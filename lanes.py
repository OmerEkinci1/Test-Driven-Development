import cv2
import numpy as np
import matplotlib.pyplot as plt 

#şeritlerin koordinatlarını yakaldım
def make_coordinates(image, line_parameters):
    slope , intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# bu fonksiyonla bulduğumuz çizgilerledeki sapmaları düzelttim.
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters= np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # sağ ve sol şetirlerdeki kesişimleri yakaladım.
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])

## görüntüyü daha net algılamak için griye çevirdik, böylelikle çizgileri daha rahat yakalayabileceğiz.Siha gibi.
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ## görüntüyü canny e çevirerek resim bölümlerini daha net seçebiliriz.Gradient Mode
    canny = cv2.Canny(blur, 50, 150)
    return canny

# hougla belirlediğimiz line larla bu fonksiyonda şeritleri yakalıyorum.
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1,y1),(x2 ,y2), (255,0,0), 10)
    return line_image

##çizgilerimizi yakalmaya çalıştığımız yeri bir polygon içine aldım.region of interest ile yol olmayan yerleri siliyorum.
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    # bitwise ile resimde sadece istediğim kısımları çalıştırdım.
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

# image = cv2.imread('Resources/test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# crooped_image = region_of_interest(canny_image)
# # hough transform ile matematiksel olarak açıklanabilen tüm şekilleri gösterebilmemiz için.Şeritte bir doğrudur.
# lines = cv2.HoughLinesP(crooped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # çizgileri gösterir.
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines) #koordinatları verir
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # resmin üzerinde çizgileri gösterir.
# cv2.imshow("result",combo_image)
# cv2.waitKey(0)
#çizgileri görebilmek(çizebilmek için) matplotlib kullandım.
#plt.imshow(canny)
#plt.show()

cap = cv2.VideoCapture("Resources/test2.mp4")
while(cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    crooped_image = region_of_interest(canny_image)
    # hough transform ile matematiksel olarak açıklanabilen tüm şekilleri gösterebilmemiz için.Şeritte bir doğrudur.
    lines = cv2.HoughLinesP(crooped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # çizgileri gösterir.
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines) #koordinatları verir
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # resmin üzerinde çizgileri gösterir.
    cv2.imshow("result",combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()