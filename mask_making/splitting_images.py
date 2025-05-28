import os

# Caminhos das três pastas
original_folder = 'data/To_Remask_Images'
ash_folder = 'data/To_Remask_Images_Ash'
maria_folder = 'data/To_Remask_Images_Maria'

# Função auxiliar para listar nomes de arquivos (sem caminho)
def list_images(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

# Listar arquivos
original_images = set(list_images(original_folder))
ash_images = set(list_images(ash_folder))
maria_images = set(list_images(maria_folder))

# Interseção (arquivos em comum entre Ash e Maria)
intersection = ash_images.intersection(maria_images)

# União das duas pastas
union = ash_images.union(maria_images)

# Comparação
missing_in_union = original_images - union
extra_in_union = union - original_images

# Resultados
print("✅ Validation Report:")
print(f"- Total in original folder: {len(original_images)}")
print(f"- Total in Ash's folder:    {len(ash_images)}")
print(f"- Total in Maria's folder:  {len(maria_images)}")

# 1. Checar interseção
if intersection:
    print(f"❌ Error: {len(intersection)} image(s) appear in both folders:")
    for img in sorted(intersection):
        print(f"  - {img}")
else:
    print("✅ No images appear in both folders.")

# 2. Checar se algo ficou de fora ou a mais
if missing_in_union:
    print(f"❌ Error: {len(missing_in_union)} image(s) missing from the combined split:")
    for img in sorted(missing_in_union):
        print(f"  - {img}")
elif extra_in_union:
    print(f"❌ Error: {len(extra_in_union)} unexpected image(s) found in the split folders:")
    for img in sorted(extra_in_union):
        print(f"  - {img}")
else:
    print("✅ All images from the original folder are accounted for — and no extras.")

