

# распознание номеров
'''
import cv2
import imutils
import numpy
import numpy as np
from matplotlib import pyplot as pl

# Получаем изображение
# img = cv2.imread('images/number2.jpg')
img = cv2.imread('images/number1.jpg')
# img = cv2.imread('images/car2.jpg')
# img = cv2.imread('images/car_men.jpg')

# Преобразование в черно-белый
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

# Фильтрация изображения
img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
# обозначение границ изображения
edges = cv2.Canny(img_filter, 55, 200)

# Поиск контуров
img_contur = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contur = imutils.grab_contours(img_contur)
img_contur = sorted(img_contur, key=cv2.contourArea, reverse=True)

# Узнаем координати номера
position = None
for c in img_contur:
    approx = cv2.approxPolyDP(c, 10, True)
    if len(approx) == 4:
        position = approx
        break

# Создаем маску для вырезания номера
mask = numpy.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [position], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

# вырезаем номерной знак
x, y = np.where(mask == 255)
x1, y1 = np.min(x), np.min(y)
x2, y2 = np.max(x), np.max(y)
crop = gray[x1:x2, y1:y2]

print(position)

pl.imshow(cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB))
pl.show()'''

# распознание лиц
'''import cv2

img = cv2.imread('images/peoples.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('models/faces.xml')               # открить файл как модель

results = faces.detectMultiScale(img_gray, scaleFactor=2.4, minNeighbors=3)     # scaleFactor размер лиц, minNeighbors количество соседей
print(results)

for (x, y, w, h) in results:        # рисуем обводку по координатам полученим в results
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

cv2.imshow("res", img)
cv2.waitKey(0)'''

# распознание лиц с вебки
'''
import cv2

# 1 захват вебки
cap = cv2.VideoCapture(0)

# 2 перебераес изображения с видео
while True:
    success, img = cap.read()

    # 3 отзеркаливание изображения
    img = cv2.flip(img, 1)

    # 4 задание размера изображения
    img = cv2.resize(img, (img.shape[1], img.shape[0]))

    # 5 создание серого слоя
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 6 подгрузка модели распознавания
    faces = cv2.CascadeClassifier('models/faces.xml')

    # 7 распознавание серого слоя, получение кординат и размеров лиц
    results = faces.detectMultiScale(img_gray, scaleFactor=2.4, minNeighbors=3)

    print(results)

    # 8 рисуем в цветном слое обводку по координатам полученим в results
    for (x, y, w, h) in results:  
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    # 9 отображение изображения
    cv2.imshow('res', img)
    print(img.shape)

    # условия завершения программы
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
'''

# побитовие операции
'''import cv2
import numpy

img = numpy.zeros((350, 350), dtype='uint8')
h, w = img.shape[:2]
img_circle = cv2.circle(img.copy(), (0, 0), 150, 255, -1)
img_square = cv2.rectangle(img.copy(), (w//6, h//6), (w//6*5, h//6*5), 255, -1)

img = cv2.bitwise_and(img_circle, img_square)
# img = cv2.bitwise_not(img_circle)       #инверсия изображения
# img = cv2.bitwise_or(img_circle, img_square)
# img = cv2.bitwise_xor(img_circle, img_square)

cv2.imshow('res', img)
cv2.imshow('res_circle', img_circle)
cv2.imshow('res_square', img_square)
cv2.waitKey(0)'''

# маски
'''import cv2
import numpy

img_car = cv2.imread('images/car2.jpg')
img = numpy.zeros(img_car.shape[:2], dtype='uint8')

h, w = img_car.shape[:2]

img_circle = cv2.circle(img.copy(), (850, 550), 150, 255, -1)
img_square = cv2.rectangle(img.copy(), (w//6, h//6), (w//6*5, h//6*5), 255, -1)

img = cv2.bitwise_and(img_car, img_car, mask=img_circle)


cv2.imshow("res", img)
cv2.waitKey(0)'''

# преобразование изображения в цветовие гаммы
'''import cv2

img = cv2.imread('images/car2.jpg')

# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
res_img = cv2.resize(img, (w//2, h//2), cv2.INTER_NEAREST)

b, g, r = cv2.split(res_img)
merge_img = cv2.merge([b, g, r])

cv2.imshow('res_b', b)
cv2.imshow('res_g', g)
cv2.imshow('res_r', r)
cv2.imshow('res', res_img)
cv2.imshow('res_merge', merge_img)
cv2.waitKey(0)'''

# отрисовка изображения по контурам
'''
import cv2
import numpy

# 1 откриваем изображене
img = cv2.imread('images/car2.jpg')

# 2 создаем новое избражение на основе данних старого(размеры и количество слоев
new_img = numpy.zeros(img.shape, dtype='uint8')


# img = cv2.flip(img, 1)                # отзеркалить  1 - горизонталь 0 - вертикаль  -1 - горизонт+вертикаль

# вращение картинки
def rotate(img_param, angle):                 # angle -угол вращения
    height, width = img.shape[:2]       # получаем срез первых 2 параметров .shape(висота, ширана, количество слоев)
    point = (width//2, height//2)       # точка вращения

    mat = cv2.getRotationMatrix2D(point, angle, 1)  # матрица с параметрами вращения, вращения, угол поворота, увеличение
    return cv2.warpAffine(img_param, mat, (width, height))
# img = rotate(img, 45)

# смещение изображения
def transform(img_param, x, y):
    mat = numpy.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img_param, mat, (img_param.shape[1], img_param.shape[0]))
# img = transform(img, 100, 300)

# 3 переводим изображение в серое и добавляем размытие для сглаживания углов
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)

# 4 получаем контуры изображения
img = cv2.Canny(img, 55, 200)     # установление прогов серого все цвета 0-55 будут 0(черный), 200-255 будут 255(белый), 55-200 будет 0 или 1 в зависимости от контекста

# 5 Ищем контур
con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)       #con - СПИСОК ВСЕХ КОНТУРОВ, hir - иеархия обектов, что в че расположено,  RETR_LIST- дает все доступные конуры, CHAIN_APPROX_NONE- координтаы всех точек контура, CHAIN_APPROX_SIMPLE - координкаты крайних точек контура

# 6 создаем изображение на основе полученных контуров
cv2.drawContours(new_img, con, -1, (255, 0, 0), 1)

# print(con)
cv2.imshow('res', img)
cv2.imshow('res_contur', new_img)
cv2.waitKey(0)'''

# 3 Обработка видео с вебки
'''import cv2
import numpy

# 1 открываем видео
cap = cv2.VideoCapture(0)

# 2 перебираем по кадрам
while True:
    success, img = cap.read()
    # cv2.imshow('res', img)

    img = cv2.flip(img, 1) # отзеркаливание
    # 3 Изменение размера изображения
    # img = cv2.resize(img, (img.shape[1] * 5//4, img.shape[0] * 5//4))  # размеры картинки
    img = cv2.resize(img, (img.shape[1], img.shape[0]))

    # 4 Преобразование цвета изображения
    # img = cv2.GaussianBlur(img, (3, 3), 0)       # размыть картинку (3, 3(только не четние)), 0(умножитель размытия))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # из цветной в серую Из RGD(BGR) в2 серую (GRAY)

    # 5 Оператор Canny используется для выделения границ на изображении
    img = cv2.Canny(img, 70, 120)  # Изображение в бинарний формато

    # 6 Детализация
    kernel = numpy.ones((2, 2), numpy.uint8)  # kernel матрица 5 на 5  numpy.uint8 целые числа больше нуля
    img = cv2.dilate(img, kernel, iterations=3)  # Задать ширину обводки iterations коэфициент
    img = cv2.erode(img, kernel, iterations=3)  # уменьшаем тольщины обрисовки (повышаем четкость)

    # 8 Отображение изображения
    cv2.imshow('res', img)  # обрезать картинку
    # cv2.imshow('res', img)
    print(img.shape)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break'''

# 3 Обработка видео
'''import cv2
import numpy

# 1 открываем видео
cap = cv2.VideoCapture('videos/number.mp4')

# 2 перебираем по кадрам
while True:
    success, img = cap.read()
    # cv2.imshow('res', img)

    # 3 Изменение размера изображения
    img = cv2.resize(img, (img.shape[1] * 5//4, img.shape[0] * 5//4))  # размеры картинки
    # img = cv2.resize(img, (400, 300))

    # 4 Преобразование цвета изображения
    # img = cv2.GaussianBlur(img, (3, 3), 0)       # размыть картинку (3, 3(только не четние)), 0(умножитель размытия))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # из цветной в серую Из RGD(BGR) в2 серую (GRAY)

    # 5 Оператор Canny используется для выделения границ на изображении
    img = cv2.Canny(img, 40, 40)  # Изображение в бинарний формато

    # 6 Дилатация
    kernel = numpy.ones((2, 2), numpy.uint8)  # kernel матрица 5 на 5  numpy.uint8 целые числа больше нуля
    img = cv2.dilate(img, kernel, iterations=2)  # Задать ширину обводки iterations коэфициент
    img = cv2.erode(img, kernel, iterations=2)  # уменьшаем тольщины обрисовки (повышаем четкость)

    # 8 Отображение изображения
    cv2.imshow('res', img[800:1400, 0:900])  # обрезать картинку
    # cv2.imshow('res', img)
    print(img.shape)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
'''

# 2 создание своего изображения текст фигуры
'''
import cv2
import numpy as np

photo = np.zeros((400, 500, 3), dtype='uint8')          # создане матрици 400 но 500 с "0", 3 - количество слоев (RGB)
                                                        # черный квадрат 400 на 500 пикселей

# photo[:] = 255, 0, 0                                    # cиний квадрат 400 на 500 пикселей (BGR)
# photo[10:150, 170:210] = 217, 21, 7                     # закрашена часть

cv2.rectangle(photo, (50, 70), (170, 200), (217, 21, 7), thickness=2)  # рисуем прямоугольник с (50, 70) по (170, 200), цветом (217, 21, 7), толщиной 2

# cv2.line(photo, (30, 30), (130, 180), (0, 0, 200), thickness=3)         # рисуем линию
cv2.line(photo, (photo.shape[0]//2, photo.shape[1]//2), (photo.shape[0], photo.shape[1]), (0, 0, 200), thickness=2)         # рисуем красную линию
cv2.line(photo, (photo.shape[1]//2, photo.shape[0]//2), (photo.shape[1], photo.shape[0]), (200, 0, 0), thickness=2)         # рисуем синюю линию
cv2.line(photo, (240, 200), (490, 400), (0, 200, 000), thickness=2)         # рисуем зеленую линию

cv2.circle(photo, (photo.shape[1]//2, photo.shape[0]//2), 50, (200, 0, 0), thickness=2)             # рисуем круг
cv2.circle(photo, (photo.shape[1]//4*3, photo.shape[0]//4*3), 30, (0, 200, 0), thickness=cv2.FILLED)             # рисуем круг

# пишем текст       текст    коорд              шрифт        коэф.увел  цвет        толщина
cv2.putText(photo, 'Text1', (200, 50), cv2.FONT_HERSHEY_TRIPLEX, 2/4, (255, 0, 0), thickness=1)
cv2.putText(photo, 'Text2', (200, 85), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=1)
cv2.putText(photo, 'Text3', (200, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)

cv2.imshow('res_photo', photo)
cv2.waitKey(0)
'''

# 1 отображение вивиод и кореция изображений
'''
import cv2
import numpy

# откритие фото
# img = cv2.imread('images/zero.jpg')
# cv2.imshow('zero_result', img)
#
# cv2.waitKey(0)



# откритие видео
# cap = cv2.VideoCapture('videos/number.mp4')
# cap.set(3, 500)    #3 это id параметра ширина
# cap.set(4, 300)    #4 это id параметра висоты
#
# while True:
#     success, img = cap.read()
#     cv2.imshow('video_rez', img)
#
#     if cv2.waitKey(10) &  0xFF == ord('q'):
#         break


#
# # откритие фото через вебку
# cap = cv2.VideoCapture(0)
# cap.set(3, 400)  # 3 это id параметра ширина
# cap.set(4, 500)  # 4 это id параметра висоты
#
# while True:
#     success, img = cap.read()
#     cv2.imshow('video_rez', img)
#
#     if cv2.waitKey(10) &  0xFF == ord('q'):
#         break


"""
для перевода изображения в бинарный формат 
1. открываем изображение
2. задаем размер картинки
3. переводим изображение из цветной в серую Из RGD(BGR) в2 серую (GRAY)
4. переводим в бинарний формат
5. задем обводку бинарним частичкам для слияния отдельных фрагментов в целое
6. уменьшаем толщину обводки уже соединенных частиц
7. выводим обрезанную часть изображения
"""

# 1
img = cv2.imread('images/car2.jpg')

# 2
img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))    # размеры картинки
# img = cv2.resize(img, (400, 300))

# 3
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # из цветной в серую Из RGD(BGR) в2 серую (GRAY)
# img = cv2.GaussianBlur(img, (3, 3), 0)       # размыть картинку (3, 3(только не четние)), 0(умножитель размытия))

# 4
img = cv2.Canny(img, 200, 200)                   # Изображение в бинарний формато

# 5
kernel = numpy.ones((5, 5), numpy.uint8)        # kernel матрица 5 на 5  numpy.uint8 целые числа больше нуля
img = cv2.dilate(img, kernel, iterations=1)     # Задать ширину обводки iterations коэфициент

# 6
img = cv2.erode(img, kernel, iterations=1)       # уменьшаем тольщины обрисовки (повышаем четкость)

# 7
cv2.imshow('res_car_men', img[900:1300, 1400:2200])    # обрезать картинку
# cv2.imshow('res_car_men', img)

print(img.shape)
cv2.waitKey(0)
'''



