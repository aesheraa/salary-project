

#yapay zeka kullnarark HR departman� maa� skalas� hesaplama projesi 


#�lk �nce hangi makine ��renimi modelini kullanaca��ma karar vermem gerekiyor. Bunun i�in veri setini iyice inceliyorum.

#Bu projede kulland���m veri setine Linear regression uyguland���mda anlaml� bir tahmin do�rusu elde edemiyorum. 
#Minimum MSE  de�erine ra�men ortaya i�e yaramaz bir sonu� ��k�yor. 
#Do�ru yapay zeka modelini kullanmak i�in Polynimial Linear regression deniyorum ve direkt de�erler grafi�ime fit ediyor.

# Polynomial Linear Regression Genel Form�l�:
# y = a + b1*x + b2*x^2 + b3*x^3 + b4*x^4 + ....... + bN*x^N

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Veri setimi dataframe'e ekliyorum csv ile. Datasette sadece deneyim ve maa� bilgisi var. Polinom fonksiyonunu �a��r�rken polinomun derecesini belirtiyorum. 
#X y ekseni belirleyerek grafi�i �izdiriyorum. De�erleri fonksiyona uyacak �ekilde uyar�l�yorum. Yani 1, x, x^2 (N=2)


# Veri setimizi pandas yard�m�yla al�p dataframe nesnemiz olan df'in i�ine aktar�yoruz..
df = pd.read_csv("polynomial.csv",sep = ";")


# Veri setimize bir bakal�m
plt.scatter(df['deneyim'],df['maas'])
plt.xlabel('Deneyim (y�l)')
plt.ylabel('Maa�')
plt.savefig('1.png', dpi=300)
plt.show()


# G�r�ld��� gibi do�rusal bir yap�da da��lm�yor veriler
# E�er biz bu veri setine linear regression uygularsak hi� uygun olmayan bir tahmin �izigisi g�r�r�z:

#Reg nesnesi olu�turup fit metodunu �a��rarak regresyon modelini mevcut verilerle e�itiyorum. 

reg = LinearRegression()
reg.fit(df[['deneyim']],df['maas'])

plt.xlabel('Deneyim (y�l)')
plt.ylabel('Maa�')

plt.scatter(df['deneyim'],df['maas'])   

xekseni = df['deneyim']
yekseni = reg.predict(df[['deneyim']])
plt.plot(xekseni, yekseni,color= "green", label = "linear regression")
plt.legend()
plt.show()

#�lk �nce hangi makine ��renimi modelini kullanaca��ma karar vermem gerekiyor. Bunun i�in veri setini iyice inceliyorum.

#Bu projede kulland���m veri setine Linear regression uyguland���mda anlaml� bir tahmin do�rusu elde edemiyorum. 
#Minimum MSE  de�erine ra�men ortaya i�e yaramaz bir sonu� ��k�yor. 

#Tahmin i�in �ok k�t� bir do�ru, demek ki neymi�: Bu veri seti i�in lineer regresyon uygulamak do�ru de�ilmi�. Unutmay�n veri setinize g�re model se�ece�iz
# �ncelikle veri setinize �ok iyi hakim olmal� ve bilmelisiniz !!!


#Bu veri seti i�in regression �e�itlerinden polynomial regression uygulanmas� gerekti�ine kara verdik. �imdi nas�l uygulad���m�za bakal�m:
# x de�erimizi polinom yukardaki fonksiyonuna uyacak �ekilde uyarlanmas�n� sa�l�yoruz

# Yani => 1, x, x^2 (N=2) �eklinde


# bir adet polynomial regression nesnesi olu�turmas� i�in PolynomialFeatures fonksiyonunu �a��r�yoruz
# Bu fonksiyonu �a��r�rken polinomun derecesini (N) belirtiyoruz:
polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(df[['deneyim']])



# regression model nesnemizi olan reg nesnemizi olu�turup bunun fit metonu �a��rarak x_polynomial ve y eksenlerini fit ediyor
# yani regresyon modelimizi mevcut ger�ek verilerle e�itiyoruz:
reg = LinearRegression()
reg.fit(x_polynomial,df['maas'])


#sss Art�k modelimiz haz�r ve e�itilmi�, �imdi eldeki verilere g�re modelimiz nas�l bir sonu� grafi�i olu�turuyor onu g�relim:


y_head = reg.predict(x_polynomial)
plt.plot(df['deneyim'],y_head,color= "red",label = "polynomial regression")
#plt.plot(xekseni, yekseni,color= "green", label = "linear regression")
plt.legend()

#veri setimizi de noktla� olarak scatter edelim de g�relim bakal�m uymu� mu polynomial regression:
plt.scatter(df['deneyim'],df['maas'])   

plt.show()


# G�rd���n�z gibi kesinlikle uymu� diyebiliriz, polynomial regression do�ru bir se�im.
# �imdi bir de N=3 veya 4 yap�p g�relim polinom derecesini art�rd���m�zda daha g�zel fit edecek mi acaba?

#Algoritmayi test etmek i�in bir ara y�netici tan�ml�yorum, seviyesi region manager ile country manager aras�nda olsun yani 4.5 . 
#Test i�in Python ile bu ara y�neticinin maa��n� �irket politikalar�na en uygun �ekilde tespit ediyorum.

x_polynomial1 = polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)



