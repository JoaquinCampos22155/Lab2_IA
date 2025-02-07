import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

def load_image_grayscale(path):
    """
    Carga una imagen en escala de grises y la devuelve 
    como un array numpy de 2D (alto x ancho).
    """
    img = Image.open(path).convert('L')  
    return np.array(img)

def split_into_blocks(img_array, C):
    """
    Subdivide la imagen (H x W) en bloques de tamaño CxC.
    Devuelve un array de bloques, cada uno de forma (C, C).
    Si H o W no son múltiplos de C, se recorta la imagen para considerar solo bloques completos.
    """
    H, W = img_array.shape
    # Recortar la imagen para que H y W sean múltiplos de C
    H_new = H - (H % C)
    W_new = W - (W % C)
    img_cropped = img_array[:H_new, :W_new]

    blocks = []
    for i in range(0, H_new, C):
        for j in range(0, W_new, C):
            block = img_cropped[i:i+C, j:j+C]
            blocks.append(block)
    return np.array(blocks)  

def vectorize_blocks(blocks):
    """
    Dado un array de bloques de forma (n_blocks, C, C),
    los aplana en vectores 1D de longitud C^2.
    Resultado: array 2D, de forma (n_blocks, C^2).
    """
    n_blocks, C, _ = blocks.shape
    return blocks.reshape(n_blocks, C*C)

def reshape_blocks(block_vectors, C):
    """
    Reconstruye los bloques vectorizados de forma (n_blocks, C^2)
    a su forma (n_blocks, C, C).
    """
    n_blocks = block_vectors.shape[0]
    return block_vectors.reshape(n_blocks, C, C)

def recombine_blocks(blocks, H, W, C):
    """
    Toma un array de bloques (n_blocks, C, C)
    y los coloca secuencialmente para recrear la imagen de tamaño H x W.
    Asume que H y W son múltiplos de C.
    """
    out_img = np.zeros((H, W), dtype=blocks.dtype)
    idx = 0
    for i in range(0, H, C):
        for j in range(0, W, C):
            out_img[i:i+C, j:j+C] = blocks[idx]
            idx += 1
    return out_img

def pca_compress_decompress(img_array, C=8, k=10):
    """
    Aplica compresión PCA por bloques CxC con n_components=k.
    Retorna la imagen reconstruida, y el MSE.
    """
    # 1) Dividir en bloques CxC
    H, W = img_array.shape
    H_new = H - (H % C)
    W_new = W - (W % C)
    img_cropped = img_array[:H_new, :W_new]
    blocks = split_into_blocks(img_cropped, C)

    # 2) Vectorizar cada bloque
    block_vectors = vectorize_blocks(blocks)  

    # 3) Ajustar PCA con n_components=k
    pca = PCA(n_components=k)
    reduced = pca.fit_transform(block_vectors)      
    reconstructed = pca.inverse_transform(reduced)  

    # 4) Reconstruir bloques
    reconstructed_blocks = reshape_blocks(reconstructed, C)  

    # 5) Combinar los bloques en la imagen final
    reconstructed_image = recombine_blocks(reconstructed_blocks, H_new, W_new, C)

    # 6) Calcular MSE 
    mse = np.mean((img_cropped - reconstructed_image)**2)

    return reconstructed_image, mse

if __name__ == "__main__":
    image_paths = ["arbol.jpg", "leon.jpg", "loros.png"]  # Lista de imágenes
    ks = [1, 5, 15, 64]  # Valores de k a probar
    C = 10  # Tamaño de bloque

    for path in image_paths:
        original = load_image_grayscale(path)  

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Resultados para {path}")

        plt.subplot(1, len(ks) + 1, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        for j, k in enumerate(ks):
            reconstructed, mse = pca_compress_decompress(original, C=C, k=k)

            plt.subplot(1, len(ks) + 1, j + 2)
            plt.imshow(reconstructed, cmap='gray')
            plt.title(f"k={k}\nMSE={mse:.2f}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()  
