import os
import cv2
import albumentations as A
from albumentations.core.composition import OneOf
import argparse


def augment_images(input_dir, output_dir, num_augmented_per_image=5):

    # Garantir que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Transformações para augmentação, você pode encontrar outras possibilidades na documentação
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.5),
        A.Rotate(limit=45, p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ])
    
    # Iterar sobre as imagens no diretório
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  

        # Ler a imagem
        image = cv2.imread(file_path)
        if image is None:
            print(f"Erro ao carregar a imagem: {file_name}")
            continue

        # Aplicar augmentações e salvar as novas imagens na pasta definida
        for i in range(num_augmented_per_image):
            augmented = transform(image=image)
            augmented_image = augmented["image"]
            output_file_name = f"{os.path.splitext(file_name)[0]}_aug_{i + 1}.jpg"
            output_file_path = os.path.join(output_dir, output_file_name)
            cv2.imwrite(output_file_path, augmented_image)

    print(f"Augmentação concluída! As imagens foram salvas em '{output_dir}'.")

if __name__ == "__main__":
    # Configuração dos argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Augmentação de dataset de imagens usando albumentations.")
    parser.add_argument("input_dir", type=str, help="Diretório de entrada contendo as imagens.")
    parser.add_argument("output_dir", type=str, help="Diretório de saída onde as imagens augmentadas serão salvas.")
    parser.add_argument("--num_augmented", type=int, default=5, help="Número de imagens augmentadas por imagem original (padrão: 5).")
    
    args = parser.parse_args()
    
    # Chamar a função com os argumentos fornecidos
    augment_images(args.input_dir, args.output_dir, args.num_augmented)