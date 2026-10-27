"""Запуск всего пайплайна без аргументов: python start_training.py.
Можно также нажать Run в IDE. Все пути привязаны к папке этого файла.
Зависимости из requirements.txt должны быть установлены заранее.
"""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import shutil
import traceback
import uuid
import zipfile
from datetime import datetime

# ================= НАСТРОЙКИ =================
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets" / "fundus_avseg"
MANIFEST = DATASET_DIR / "manifest.csv"
RUNS_DIR = PROJECT_DIR / "runs"

# Это выбранные настройки реконструкции, не подтверждённый mapping авторов.
# Зелёные пересечения конвертер включает в оба канала.
# background: белые пиксели -> 0 в обоих каналах (НЕ ignore).
# both: белые пиксели -> 1 в обоих каналах.
WHITE_POLICY = "background"
LOSS = "paper"  # "repository" либо "paper"; меняется LOSS, не архитектура.
BASELINE = False  # True отключает TFFM; это отдельная ablation, не режим paper.
BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95
BOOTSTRAP_SEED = 42
SEED = 42
EPOCHS = 500
PATIENCE = 10
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
INTENSITY = "minmax"
PRETRAINED = True
AUGMENTATION = True
REQUIRE_CUDA = True  # Не начинать случайно длительное обучение на CPU.

# None: новое обучение в новой папке при каждом запуске.
# Для продолжения явно укажите путь к своему доверенному last.pt.
# Пример: RESUME_CHECKPOINT = PROJECT_DIR / "runs" / "имя_запуска" / "last.pt"
RESUME_CHECKPOINT = None
PAUSE_ON_EXIT = os.name == "nt"

# После успешного обучения: оценка best.pt на test и два отдельных ZIP.
# Test не участвует в выборе checkpoint: best.pt выбирается по val loss.
EVALUATE_TEST_AFTER_TRAINING = True
CREATE_ARCHIVES = True
ARCHIVES_DIR = RUNS_DIR / "archives"
# ============================================


def run_step(title, script, arguments):
    print("\n" + "=" * 65, flush=True)
    print(title, flush=True)
    print("=" * 65, flush=True)
    subprocess.run(
        [sys.executable, "-u", str(PROJECT_DIR / script), *map(str, arguments)],
        cwd=str(PROJECT_DIR),
        check=True,
    )


def write_json_file(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def export_history_csv(out):
    path = out / "history.json"
    if not path.is_file():
        return
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        return
    fields = list(dict.fromkeys(key for row in history for key in row))
    target = out / "history.csv"
    tmp = target.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    tmp.replace(target)


def make_zip(target, items):
    # Исходные результаты не удаляются. Неполный ZIP не получает финальное имя.
    tmp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as archive:
            for path, name in items:
                archive.write(path, arcname=name)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"Архив готов: {target} ({target.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def package_results(out, status):
    out.mkdir(parents=True, exist_ok=True)
    write_json_file(out / "launch_status.json", status)
    export_history_csv(out)
    snapshot = out / "source"
    snapshot.mkdir(exist_ok=True)
    # Сохраняем также настройки запуска и преобразования данных.
    for name in ("start_training.py", "prepare_avseg.py"):
        source = PROJECT_DIR / name
        if source.is_file():
            shutil.copy2(source, snapshot / name)
    report = DATASET_DIR / "conversion_report.json"
    if report.is_file():
        shutil.copy2(report, out / "conversion_report.json")
    if not CREATE_ARCHIVES:
        return
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{out.name}_{stamp}"
    weight_suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
    # Не включаем датасет, вероятности/предсказания и незавершённые файлы.
    # Веса не могут попасть в архив метрик даже из вложенных папок.
    reports, weights = [], []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(out)
        if "predictions" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix.lower() in {".tmp", ".zip", ".pyc"}:
            continue
        item = (path, (Path(out.name) / relative).as_posix())
        if path.suffix.lower() in weight_suffixes:
            weights.append(item)
        else:
            reports.append(item)
    make_zip(ARCHIVES_DIR / f"{prefix}_metrics_history.zip", reports)
    if weights:
        make_zip(ARCHIVES_DIR / f"{prefix}_weights.zip", weights)
    else:
        print("Сохранённых checkpoints нет: архив весов не создаётся.", flush=True)


def prepare_data():
    # Существующий split всегда сохраняется, даже если SEED изменён.
    if MANIFEST.is_file():
        print(f"Сохраняю существующее разбиение: {MANIFEST}", flush=True)
        print("Маски не конвертируются повторно; WHITE_POLICY к ним не применяется.", flush=True)
        return
    if RESUME_CHECKPOINT is not None:
        raise RuntimeError(
            "Для resume требуется исходный manifest.csv. Восстановите его из "
            "папки предыдущего запуска; новое разбиение автоматически не создаётся."
        )
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    samples = DATASET_DIR / "samples.csv"
    report_path = DATASET_DIR / "conversion_report.json"
    if samples.is_file():
        # samples.csv мог быть подготовлен пользователем с собственными масками.
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("white_policy") != WHITE_POLICY:
                raise RuntimeError(
                    "WHITE_POLICY отличается от политики уже созданных масок. "
                    "Согласуйте настройку с conversion_report.json либо используйте "
                    "отдельную папку данных для новой конвертации."
                )
        else:
            print("Используется ваш samples.csv; mapping масок берётся как есть.", flush=True)
    else:
        partial = [p for p in (DATASET_DIR / "arteries", DATASET_DIR / "veins", report_path) if p.exists()]
        if partial:
            raise RuntimeError(
                "Найдены результаты неполной подготовки без samples.csv: "
                + ", ".join(map(str, partial))
                + ". Проверьте их вручную; автоматически ничего не удаляется."
            )
        run_step("1. Проверка и конвертация 100 RGB-масок", "prepare_avseg.py", [
            "--root", DATASET_DIR, "--white-policy", WHITE_POLICY,
        ])
    run_step("2. Создание нового разбиения 80/10/10", "run.py", [
        "split", "--input", samples, "--output", MANIFEST, "--seed", SEED,
    ])


def main():
    required = ["run.py", "data.py", "model.py", "loss.py", "paper_losses.py", "metrics.py", "interval_metrics.py"]
    if not MANIFEST.is_file() and not (DATASET_DIR / "samples.csv").is_file():
        required.append("prepare_avseg.py")
    missing = [name for name in required if not (PROJECT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Положите рядом со start_training.py файлы: " + ", ".join(missing))
    if WHITE_POLICY not in ("background", "both", "error") or LOSS not in ("repository", "paper"):
        raise ValueError("Неверные WHITE_POLICY или LOSS в настройках.")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch не установлен в выбранном интерпретаторе. "
            "Установите зависимости requirements.txt и выберите это окружение в IDE."
        ) from exc
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна в выбранном Python. Проверьте драйвер NVIDIA и CUDA-сборку "
            "PyTorch. Для намеренного обучения на CPU задайте REQUIRE_CUDA = False."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Python: {sys.executable}", flush=True)
    print(f"Устройство: {device}", flush=True)
    print(f"Данные: {DATASET_DIR}", flush=True)
    prepare_data()
    # read_manifest в train дополнительно проверит пути, split и утечки групп.
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    counts = {s: sum(r.get("split") == s for r in rows) for s in ("train", "val", "test")}
    print(f"Изображений по split: {counts}", flush=True)
    if not all(counts.values()):
        raise ValueError("manifest.csv должен содержать непустые train, val и test.")
    if counts["train"] % BATCH_SIZE == 1:
        raise ValueError("Последний train batch содержит 1 изображение. Измените BATCH_SIZE.")
    resume = None
    if RESUME_CHECKPOINT is not None:
        resume = Path(RESUME_CHECKPOINT)
        if not resume.is_absolute():
            resume = PROJECT_DIR / resume
        resume = resume.resolve()
        if not resume.is_file():
            raise FileNotFoundError(resume)
        out = resume.parent
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prefix = "baseline" if BASELINE else "tffm"
        out = RUNS_DIR / f"{prefix}_{LOSS}_{stamp}"
    arguments = [
        "train", "--manifest", MANIFEST, "--out", out,
        "--device", device, "--loss", LOSS, "--seed", SEED,
        "--epochs", EPOCHS, "--patience", PATIENCE, "--batch-size", BATCH_SIZE,
        "--lr", LEARNING_RATE, "--weight-decay", WEIGHT_DECAY,
        "--intensity", INTENSITY,
        "--pretrained" if PRETRAINED else "--no-pretrained",
    ]
    if BASELINE:
        arguments.append("--baseline")
    if not AUGMENTATION:
        arguments.append("--no-augmentation")
    if resume:
        arguments.extend(["--resume", resume])
    print(f"Результаты будут в: {out}", flush=True)
    print(f"Loss: {LOSS}; TFFM включён: {not BASELINE}", flush=True)
    print("ВНИМАНИЕ: это реконструкция. Seed, mapping меток и ряд деталей авторов не подтверждены.", flush=True)
    status = {
        "started_at": datetime.now().isoformat(),
        "training": "running",
        "test_evaluation": "not_started" if EVALUATE_TEST_AFTER_TRAINING else "disabled",
        "output_directory": str(out),
        "resume_checkpoint": str(resume) if resume else None,
    }
    try:
        run_step("3. Обучение TFFM", "run.py", arguments)
        status["training"] = "completed"
        if EVALUATE_TEST_AFTER_TRAINING:
            best = out / "best.pt"
            if not best.is_file():
                raise FileNotFoundError(f"Нельзя оценить модель: отсутствует {best}")
            status["test_evaluation"] = "running"
            # Отдельная папка для каждой оценки исключает смешивание с прошлой.
            evaluation_dir = out / ("test_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            status["test_output_directory"] = str(evaluation_dir)
            run_step("4. Метрики best.pt на отложенном test", "run.py", [
                "evaluate", "--checkpoint", best, "--manifest", MANIFEST,
                "--out", evaluation_dir, "--device", device,
                "--batch-size", BATCH_SIZE, "--split", "test",
                "--bootstrap-samples", BOOTSTRAP_SAMPLES,
                "--confidence", CONFIDENCE, "--bootstrap-seed", BOOTSTRAP_SEED,
            ])
            status["test_evaluation"] = "completed"
    except BaseException as exc:
        if status["training"] == "running":
            status["training"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        elif EVALUATE_TEST_AFTER_TRAINING:
            status["test_evaluation"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        status["finished_at"] = datetime.now().isoformat()
        try:
            print("\n5. Сохранение отчётов и создание отдельных архивов", flush=True)
            package_results(out, status)
        except Exception:
            # Ошибка архивирования не должна скрыть исходную ошибку обучения.
            print("Не удалось завершить упаковку. Исходные результаты остаются в папке запуска.", flush=True)
            traceback.print_exc()
            if not active_error:
                raise
    print(f"\nОбучение завершено. Результаты: {out}", flush=True)
    print("Метрики/история и checkpoints упакованы раздельно (если CREATE_ARCHIVES = True).", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем. Сохранённый last.pt можно использовать для resume.", flush=True)
        exit_code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nЗапуск остановлен. См. причину ошибки выше. Данные автоматически не удаляются.", flush=True)
        exit_code = 1
    finally:
        if PAUSE_ON_EXIT and sys.stdin is not None and sys.stdin.isatty():
            try:
                input("\nНажмите Enter, чтобы закрыть окно...")
            except (EOFError, KeyboardInterrupt):
                pass
    sys.exit(exit_code)
