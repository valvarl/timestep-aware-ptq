import os
from cleanfid import fid

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Группы: {название: [(сгенерированное, референсное)]}
groups = {
    "smoothing": [
        ("output/final/output_1_2.png", "output/final/output_0_2.png"),
        ("output/final/output_2_2.png", "output/final/output_0_2.png"),
        ("output/final/output_3_2.png", "output/final/output_0_2.png"),
        ("output/final/output_4_2.png", "output/final/output_0_2.png"),
    ],
    "no_smoothing": [
        ("output/no_smoothing/output_1_2.png", "output/no_smoothing/output_0_2.png"),
        ("output/no_smoothing/output_2_2.png", "output/no_smoothing/output_0_2.png"),
        ("output/no_smoothing/output_3_2.png", "output/no_smoothing/output_0_2.png"),
        ("output/no_smoothing/output_4_2.png", "output/no_smoothing/output_0_2.png"),
    ],
    "ts_independent": [
        ("output/ts_independent/output_1_2.png", "output/ts_independent/output_0_2.png"),
        ("output/ts_independent/output_2_2.png", "output/ts_independent/output_0_2.png"),
        ("output/ts_independent/output_3_2.png", "output/ts_independent/output_0_2.png"),
        ("output/ts_independent/output_4_2.png", "output/ts_independent/output_0_2.png"),
    ]
}

def calculate_fid_for_pairs(group_name, image_pairs):
    print(f"\nFID scores for: {group_name}")
    for i, (gen, ref) in enumerate(image_pairs, start=1):
        # Создаем временные папки
        os.makedirs("tmp/gen", exist_ok=True)
        os.makedirs("tmp/ref", exist_ok=True)

        # Копируем изображения во временные папки
        os.system(f"cp {gen} tmp/gen/gen.png")
        os.system(f"cp {ref} tmp/ref/ref.png")

        # Считаем FID
        fid_score = fid.compute_fid("tmp/gen", "tmp/ref", mode="clean")
        print(f"Iteration {i}: FID = {fid_score:.4f}")

        # Очищаем временные файлы
        os.remove("tmp/gen/gen.png")
        os.remove("tmp/ref/ref.png")

if __name__ == "__main__":
    for group_name, image_pairs in groups.items():
        calculate_fid_for_pairs(group_name, image_pairs)
