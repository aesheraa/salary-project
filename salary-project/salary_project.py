

#yapay zeka kullnarark HR departmaný maaþ skalasý hesaplama projesi 


#Ýlk önce hangi makine öðrenimi modelini kullanacaðýma karar vermem gerekiyor. Bunun için veri setini iyice inceliyorum.

#Bu projede kullandýðým veri setine Linear regression uygulandýðýmda anlamlý bir tahmin doðrusu elde edemiyorum. 
#Minimum MSE  deðerine raðmen ortaya iþe yaramaz bir sonuç çýkýyor. 
#Doðru yapay zeka modelini kullanmak için Polynimial Linear regression deniyorum ve direkt deðerler grafiðime fit ediyor.

# Polynomial Linear Regression Genel Formülü:
# y = a + b1*x + b2*x^2 + b3*x^3 + b4*x^4 + ....... + bN*x^N

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Veri setimi dataframe'e ekliyorum csv ile. Datasette sadece deneyim ve maaþ bilgisi var. Polinom fonksiyonunu çaðýrýrken polinomun derecesini belirtiyorum. 
#X y ekseni belirleyerek grafiði çizdiriyorum. Deðerleri fonksiyona uyacak þekilde uyarýlýyorum. Yani 1, x, x^2 (N=2)


# Veri setimizi pandas yardýmýyla alýp dataframe nesnemiz olan df'in içine aktarýyoruz..
df = pd.read_csv("polynomial.csv",sep = ";")


# Veri setimize bir bakalým
plt.scatter(df['deneyim'],df['maas'])
plt.xlabel('Deneyim (yýl)')
plt.ylabel('Maaþ')
plt.savefig('1.png', dpi=300)
plt.show()


# Görüldüðü gibi doðrusal bir yapýda daðýlmýyor veriler
# Eðer biz bu veri setine linear regression uygularsak hiç uygun olmayan bir tahmin çizigisi görürüz:

#Reg nesnesi oluþturup fit metodunu çaðýrarak regresyon modelini mevcut verilerle eðitiyorum. 

reg = LinearRegression()
reg.fit(df[['deneyim']],df['maas'])

plt.xlabel('Deneyim (yýl)')
plt.ylabel('Maaþ')

plt.scatter(df['deneyim'],df['maas'])   

xekseni = df['deneyim']
yekseni = reg.predict(df[['deneyim']])
plt.plot(xekseni, yekseni,color= "green", label = "linear regression")
plt.legend()
plt.show()

#Ýlk önce hangi makine öðrenimi modelini kullanacaðýma karar vermem gerekiyor. Bunun için veri setini iyice inceliyorum.

#Bu projede kullandýðým veri setine Linear regression uygulandýðýmda anlamlý bir tahmin doðrusu elde edemiyorum. 
#Minimum MSE  deðerine raðmen ortaya iþe yaramaz bir sonuç çýkýyor. 

#Tahmin için çok kötü bir doðru, demek ki neymiþ: Bu veri seti için lineer regresyon uygulamak doðru deðilmiþ. Unutmayýn veri setinize göre model seçeceðiz
# Öncelikle veri setinize çok iyi hakim olmalý ve bilmelisiniz !!!


#Bu veri seti için regression çeþitlerinden polynomial regression uygulanmasý gerektiðine kara verdik. Þimdi nasýl uyguladýðýmýza bakalým:
# x deðerimizi polinom yukardaki fonksiyonuna uyacak þekilde uyarlanmasýný saðlýyoruz

# Yani => 1, x, x^2 (N=2) þeklinde


# bir adet polynomial regression nesnesi oluþturmasý için PolynomialFeatures fonksiyonunu çaðýrýyoruz
# Bu fonksiyonu çaðýrýrken polinomun derecesini (N) belirtiyoruz:
polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(df[['deneyim']])



# regression model nesnemizi olan reg nesnemizi oluþturup bunun fit metonu çaðýrarak x_polynomial ve y eksenlerini fit ediyor
# yani regresyon modelimizi mevcut gerçek verilerle eðitiyoruz:
reg = LinearRegression()
reg.fit(x_polynomial,df['maas'])


#sss Artýk modelimiz hazýr ve eðitilmiþ, þimdi eldeki verilere göre modelimiz nasýl bir sonuç grafiði oluþturuyor onu görelim:


y_head = reg.predict(x_polynomial)
plt.plot(df['deneyim'],y_head,color= "red",label = "polynomial regression")
#plt.plot(xekseni, yekseni,color= "green", label = "linear regression")
plt.legend()

#veri setimizi de noktlaý olarak scatter edelim de görelim bakalým uymuþ mu polynomial regression:
plt.scatter(df['deneyim'],df['maas'])   

plt.show()


# Gördüðünüz gibi kesinlikle uymuþ diyebiliriz, polynomial regression doðru bir seçim.
# Þimdi bir de N=3 veya 4 yapýp görelim polinom derecesini artýrdýðýmýzda daha güzel fit edecek mi acaba?

#Algoritmayi test etmek için bir ara yönetici tanýmlýyorum, seviyesi region manager ile country manager arasýnda olsun yani 4.5 . 
#Test için Python ile bu ara yöneticinin maaþýný þirket politikalarýna en uygun þekilde tespit ediyorum.

x_polynomial1 = polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)



