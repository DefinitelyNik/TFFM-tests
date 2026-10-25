from pathlib import Path
from PIL import Image

root = Path("datasets/fundus_avseg/raw")
annotations = root / "annotation"

if not (root / "images").is_dir() or not annotations.is_dir():
    raise SystemExit(
        "Не найдены raw/images и raw/annotation. "
        "Проверь вложенность папок после распаковки."
    )

files = sorted(p for p in annotations.rglob("*") if p.is_file())

print("Файлов в annotation:", len(files))
print("\nПервые 20 файлов:")
for p in files[:20]:
    print(p.relative_to(root))

checked = 0
for p in files:
    try:
        with Image.open(p) as im:
            print(f"\nМаска: {p.relative_to(root)}")
            print("Режим:", im.mode, "Размер:", im.size)

            # Для палитровых PNG проверяем реальные RGB-цвета.
            rgb = im.convert("RGB")
            colors = rgb.getcolors(maxcolors=256)

            if colors is None:
                print("Более 256 цветов — нужна дополнительная проверка.")
            else:
                colors.sort(reverse=True)
                print("Цвет RGB → количество пикселей:")
                for count, color in colors:
                    print(color, "→", count)

    except OSError:
        continue

    checked += 1
    if checked == 3:
        break

if checked == 0:
    print("\nНе удалось найти читаемые изображения масок.")