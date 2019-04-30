import cv2
from skimage import data
import numpy as np
from skimage import io, color, img_as_ubyte, util
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
import statistics
import math
from scipy.spatial import distance

#funkcja pokazująca dwa obrazy obok siebie
def show2imgs(im1, im2, title1='Obraz pierwszy', title2='Obraz drugi', size=(10,10)):
    
    import matplotlib.pyplot as plt
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=size)
    ax1.imshow(im1, cmap='gray')
    ax1.axis('off')
    ax1.set_title(title1)

    ax2.imshow(im2, cmap='gray')
    ax2.axis('off')
    ax2.set_title(title2)
    plt.show()



#funkcja wykonujaca wszystkie zadane przez nas operacje na zdjeciach
def funkcja_poc(url):
    #zaladowanie obrazu oryginalnego
    t = io.imread(url)   
    print('\n Zdjęcie: ', url, '\n')
    
    #konwersja obrazu kolorowego do formatu wieloodcieniowego
    t = color.rgb2gray(t)
    t = img_as_ubyte(t) 
    
    #wyswietlanie obrazu oryginalnego
    plt.figure(figsize=(9,12))
    plt.imshow(t, cmap="gray")
    plt.axis('off')
    plt.show()

    #filtr medianowy- usuwa zakłócenia, nie niszczy krawędzi
    mtimf = cv2.medianBlur(t,  25)

    #binaryzacja
    th = 100
    th, bim = cv2.threshold(mtimf, thresh=th, maxval=255, type=cv2.THRESH_BINARY)

    #zamkniecie obrazu
    element = np.ones((3,3),np.uint8)
    mbim = cv2.morphologyEx(bim, op=cv2.MORPH_CLOSE, kernel=element, iterations=3)
    
    #algorytm Canny do znajdowania krawedzi obiektow
    cim = cv2.Canny(mbim, 200, 250, apertureSize = 5, L2gradient = True)

    #wykrywanie obiektow (poki co polaczonych)
    fill_coins = ndi.binary_fill_holes(cim)

    #erozja celem rozdzielenia polaczonych obiektow
    kernel = np.ones((15,15),np.uint8)
    coins_cleaned = np.array(fill_coins, dtype=np.uint8)
    erodeBin = cv2.erode(coins_cleaned, kernel=kernel, iterations=6)

    #indeksacja rozdzielonych obiektow
    label_objects, nb_labels = ndi.label(erodeBin)
    
    #dylatcja - odwrocenie wczesniejszej erozji celem przywrocenia wielkosci monet
    zdj_wynik = np.array(label_objects, dtype=np.uint8)
    zdj_wynik = cv2.dilate(zdj_wynik, kernel=kernel, iterations=6)
    sizes = np.bincount(zdj_wynik.ravel())
    mask_sizes = sizes > 200
    mask_sizes[0] = 0
    
    #wyswietlanie obrazu z zaindeksowanymi obiektami
    print('Ilość monet: ', nb_labels, '\n')
    #plt.figure(figsize=(3,4))
    plt.imshow(zdj_wynik, cmap='hot')
    plt.show()
    
    #liczenie nominalow
    wielkosci = sizes[mask_sizes==True]
    ilosci_nominalow = [0, 0, 0, 0, 0]
    for n in range(len(wielkosci)):
        for m in range(len(ilosci_nominalow)):
            if m == 2: continue
            elif (wielkosci[n] > zakresy_wielkosci[m,0] and wielkosci[n] <= zakresy_wielkosci[m,1]):
                ilosci_nominalow[m] = ilosci_nominalow[m]+1
                
    print('ilosc 5 zl: ', ilosci_nominalow[0])  
    print('ilosc 1 i 2 zl: ', ilosci_nominalow[1])   
    print('ilosc 10 gr: ', ilosci_nominalow[3])   
    print('ilosc 5 gr: ', ilosci_nominalow[4], '\n')   
    
    zdj_wynik = cv2.resize(zdj_wynik,(400,300))   #przeskalowanie zdjecia celem przyspieszenia dzialania algorytmu
     
    suma = 0
    
    for i in range(nb_labels):
        print('Obiekt nr: ',i+1)
        pts = getFigure(zdj_wynik, i+1)
        print('Zajmowany % powierzchni calego obrazu: ', round((len(pts)/(zdj_wynik.shape[0]*zdj_wynik.shape[1]))*100, 2), '%')
        bb = computeBB(pts)
        feret = computeFeret(pts)
        sr_ciezkosci = cog2(pts)
        suma = suma + len(pts)
        print('Liczba punktow: ',len(pts),'\nSrodek ciezkosci: ', sr_ciezkosci,'\nBlair-Bliss: ', bb,'\nFeret: ',feret, '\n---\n')
        
    print('Zajmowany % powierzchni calego obrazu przez wszystkie obiekty :', round((suma/(zdj_wynik.shape[0]*zdj_wynik.shape[1]))*100,2), '%')
    


#funkcja do znalezienia zakresow wielkosci odpowiadajacych poszczegolnym nominalom
#funkcja wyonuje analogiczne przeksztalcenia jak funkcja funkcja_poc
def kalibracja(url):
    t = io.imread(url)
    t = color.rgb2gray(t)
    t = img_as_ubyte(t) 
    
    mtimf = cv2.medianBlur(t,  25)

    th = 100
    th, bim = cv2.threshold(mtimf, thresh=th, maxval=255, type=cv2.THRESH_BINARY)

    element = np.ones((3,3),np.uint8)
    mbim = cv2.morphologyEx(bim, op=cv2.MORPH_CLOSE, kernel=element, iterations=3)

    cim = cv2.Canny(mbim, 200, 250, apertureSize = 5, L2gradient = True)

    fill_coins = ndi.binary_fill_holes(cim)

    kernel = np.ones((15,15),np.uint8)

    coins_cleaned = np.array(fill_coins, dtype=np.uint8)
    erodeBin = cv2.erode(coins_cleaned, kernel=kernel, iterations=6)

    label_objects, nb_labels = ndi.label(erodeBin)
    
    zdj_wynik = np.array(label_objects, dtype=np.uint8)
    zdj_wynik = cv2.dilate(zdj_wynik, kernel=kernel, iterations=6)
    sizes = np.bincount(zdj_wynik.ravel())
    mask_sizes = sizes > 200
    mask_sizes[0] = 0
    
    #na przeksztalconych obrazach dokonujemy pomiaru wielkosci monet
    wielkosci = sizes[mask_sizes==True]

    sr = statistics.mean(wielkosci)
    odst = statistics.stdev(wielkosci)
    zakres = [sr-2*odst, sr+2*odst]
    return zakres


#definiujemy dodatkowe funckje umozliwiajace dalsze pomiary
def getFigure(labelledImage, objNumber):
    
    points = []
    for y in range(labelledImage.shape[0]):
        for x in range(labelledImage.shape[1]):
            if labelledImage[y,x] == objNumber:
                points.append((y,x))

    return points

#srodek ciezkosci
def cog2(points):
    mx=0
    my=0
    for (y,x) in points:
        mx = mx + x
        my = my + y
    mx = mx/len(points)
    my = my/len(points)
    
    return [my, mx]

#wspolczynnik Blaira-Blissa
def computeBB(points):
    
    s = len(points)
    my,mx = cog2(points)
    
    r = 0
    for point in points:
         r = r + distance.euclidean(point,(my,mx))**2
            
    return s/(math.sqrt(2*math.pi*r))

#wspolczynnik Fereta
def computeFeret(points):
    
    px = [x for (y,x) in points]
    py = [y for (y,x) in points]
    
    fx = max(px) - min(px)
    fy = max(py) - min(py)
    
    return float(fy)/float(fx)  


#uzywamy funkcji kalibracja 5 razy aby uzyskac zakres wielkoscimoent dla kazdego nominalu
zakresy_wielkosci = np.zeros((5,2))
nominaly_kalibracja = ['IMG_3119.JPG','IMG_3127.JPG','IMG_3134.JPG','IMG_3141.JPG','IMG_3144.JPG']
for j in range(0,5):
    zakresy_wielkosci[j,] = kalibracja(nominaly_kalibracja[j])


#dla pozostalych zdjec wywolujemy funkcje funkcja_poc ktora wykonuje wszystkie zadane operacje i pomiary
zdjecia=['IMG_3148.JPG','IMG_3158.JPG','IMG_3161.JPG','IMG_3166.JPG','IMG_3169.JPG','IMG_3172.JPG','IMG_3174.JPG',
         'IMG_3176.JPG','IMG_3180.JPG','IMG_3184.JPG','IMG_3185.JPG','IMG_3189.JPG','IMG_3190.JPG','IMG_3196.JPG',
         'IMG_3200.JPG']

for i in range(len(zdjecia)):
    x = zdjecia[i]
    funkcja_poc(x)


#segemntacja
#funkcja zwracajaca liste sasiadujacych punktow, jesli wciz sa w obrazie
def lista_kandydatow(x, y, rozmiar):
    listakand = []
    x_max = rozmiar[0]-1
    y_max = rozmiar[1]-1

    #top left
    x_new = min(max(x-1,0),x_max)
    y_new = min(max(y-1,0),y_max)
    listakand.append((x_new,y_new))

    #top center
    #x_new = x
    y_new = min(max(y-1,0),y_max)
    listakand.append((x,y_new))

    #top right
    x_new = min(max(x+1,0),x_max)
    y_new = min(max(y-1,0),y_max)
    listakand.append((x_new,y_new))

    #left
    x_new = min(max(x-1,0),x_max)
    #y_new = y
    listakand.append((x_new,y))

    #right
    x_new = min(max(x+1,0),x_max)
    #y_new = y
    listakand.append((x_new,y))

    #bottom left
    x_new = min(max(x-1,0),x_max)
    y_new = min(max(y+1,0),y_max)
    listakand.append((x_new,y_new))

    #bottom center
    #x_new = x
    y_new = min(max(y+1,0),y_max)
    listakand.append((x,y_new))

    #bottom right
    x_new = min(max(x+1,0),x_max)
    y_new = min(max(y+1,0),y_max)
    listakand.append((x_new,y_new))

    return listakand



#dwuprogowa segmentacja przez rozrost
thresh1 = 100
thresh2 = 50

def rozrost(zdj, punkt_startowy):
    lista = []
    zdj_wynik = np.zeros_like(zdj)
    lista.append((punkt_startowy[0], punkt_startowy[1]))  #punkt startowy dopisujemy do listy punktow do rozpatrzenia
    przetworzone = []
    while(len(lista) > 0):
        pkt = lista[0]
        zdj_wynik[pkt[0], pkt[1]] = 255  #przypisujemy nowa wartosc znalezionemu punktowi
        for wspolrzedne in lista_kandydatow(pkt[0], pkt[1], zdj.shape):
            if zdj[wspolrzedne[0], wspolrzedne[1]] > thresh1:  #sprawdzamy czy sasiadujace punkty sa powyzej progu 1
                zdj_wynik[wspolrzedne[0], wspolrzedne[1]] = 255  #jesli tak, to przypisujemy nowa wartosc
                if not wspolrzedne in przetworzone:
                    lista.append(wspolrzedne)
                przetworzone.append(wspolrzedne)  #dopisujemy punkt do przetworzonych aby nie rozpatrywac go ponownie
            #jesli punkt lezy miedzy progiem 2 a progiem 1 to dopisujemy go do obszaru jesli ktorys z sasiadujacych
            #punktow lezy powyzej progu 1
            elif zdj[wspolrzedne[0], wspolrzedne[1]] > thresh2 and zdj[wspolrzedne[0], wspolrzedne[1]] <= thresh1:
                control = False
                for wspolrzedne2 in lista_kandydatow(wspolrzedne[0], wspolrzedne[1], zdj.shape):
                    if zdj[wspolrzedne2[0], wspolrzedne2[1]] > thresh1:
                        control = True
                        break
                if control == True:
                    zdj_wynik[wspolrzedne[0], wspolrzedne[1]] = 255
                    if not wspolrzedne in przetworzone:
                        lista.append(wspolrzedne)
                    przetworzone.append(wspolrzedne)              
        lista.pop(0)  #usuwamy pierwsza pozycje z listy punktow do przetworzenia
    return zdj_wynik


#indeksowanie obszarow wykrytych przez funkcje rozrost
start = 90

def indeksacja(zdj):
    #lista punktow startowych
    lista_start = []
    lista_obszarow = []
    indeks = 1
    lista_indeksow = np.zeros_like(zdj)
    for i in range(zdj.shape[0]):   #tworzymy liste potencjalnych punktow startowych
        for j in range(zdj.shape[1]):
            if zdj[i][j] > start:
                lista_start.append([i,j]) 
    for m in range(len(lista_start)):
        #sprawdzamy czy dany punkt ma juz przypisany indeks
        if lista_indeksow[lista_start[m][0],lista_start[m][1]] == 0:
            obszar = rozrost(zdj,lista_start[m])   #jesli nie ma indeksu, wywolujemy segmentacje przez rozrost
            if (len(obszar[(obszar == 255) == True]) > 50):   #bierzemy pod uwage tylko wystarczajaco duze obszary
                lista_obszarow.append(obszar)   
                lista_indeksow[(obszar == 255) == True] = indeks   #przypisujemy nowemu obszarowi indeks
                indeks = indeks + 1   #podwyzszamy indeks
    return lista_obszarow, lista_indeksow, indeks-1


#przyklad dzialania funkcji indeksacja
url = 'IMG_3068.JPG'
p = io.imread(url)
p = color.rgb2gray(p)
p = img_as_ubyte(p) 

img = cv2.medianBlur(p,  5)

img2 = cv2.resize(img,(400,300))

#potencjalnie punkty startowe
lista3 = []

for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if p[i][j] > 90:
            lista3.append([i,j]) 
            
print(len(lista3))


obszary, indeksy, ilosc = indeksacja(img2)



print('Ilosc znalezionych obszarow: ', ilosc)
print('Wielkosci poszczegolnych obszarow:')
for n in range(len(obszary)):
    print('Obszar nr', n+1, ' - wielkosc: ', len(obszary[n][(obszary[n] == 255) == True]))



#wyswietlanie zdjecia z wyodrebnionymi obszarami
zdj_indeksowane = np.zeros_like(img2)
zdj_indeksowane[indeksy > 0] = 255
show2imgs(p, zdj_indeksowane, title1='Obraz oryginalny', title2='Obraz po indeksacji', size=(30,30))






