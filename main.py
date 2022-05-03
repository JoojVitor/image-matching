import cv2


def orb_sim(img1, img2):
    orb = cv2.ORB_create(1000)

    # Esse é o ponto onde serão dectados pontos chaves e descritores das imagens
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # Definir o objeto combinador de força bruta.
    # Isso quer dizer que ele irá testar muitas combinações diferentes.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    # Nessa etapa será feito o match das imagens, ou seja, a combinação das imagens
    matches = bf.match(desc_a, desc_b)

    # Aqui será feito a procura por regiões similares das imagens com a distância menor que 50.
    # É possível ajustar a distância entre 0 a 100, sendo 0 as imagens totalmente diferente e 100 as imagens iguais
    # Nesse caso é escolhido a distância de 50.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


img00 = cv2.imread('musk1.jpg')
img01 = cv2.imread('musk2.jpg')

img00 = cv2.resize(img00, (600, 800))
img01 = cv2.resize(img01, (600, 800))

orb_similarity = orb_sim(img00, img01)
print("Percentual de Similaridade: ", orb_similarity)

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img00, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img01, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)

matches = sorted(matches, key=lambda x: x.distance)

matched_img = cv2.drawMatches(img00, keypoints_1, img01, keypoints_2, matches[:1000], img01, flags=2)

cv2.imshow('image', matched_img)
cv2.waitKey(0)
