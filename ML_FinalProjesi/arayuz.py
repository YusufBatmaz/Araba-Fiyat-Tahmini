#Gerekli kütüphaneleri içe aktaralım
import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Bu kod satırları, daha önce kaydedilmiş olan RandomForestRegressor modelini yükler ve Streamlit kullanarak bir başlık oluşturur.
with open('model.pkl', 'rb') as f:
    model = pk.load(f)

st.header('Car Price Prediction ML Model')

# car_price_prediction_cleaned.csv adlı dosyadan veri okuyalım ve cars_data adlı bir DataFrame oluşturalım.
# Ardından, Levy sütunundaki '-' karakterlerini 0 ile değiştirir ve bu sütunu sayısal değerlere dönüştürür. 
cars_data = pd.read_csv('car_price_prediction_cleaned.csv')
cars_data['Levy'] = cars_data['Levy'].replace('-', '0')
cars_data['Levy'] = pd.to_numeric(cars_data['Levy'])

# Streamlit kullanarak bir kullanıcı arayüzü oluşturalım ve belirli bir üretici ve model seçimini sağlayalım.
manufacturer = st.selectbox('Select Manufacturer', cars_data['Manufacturer'].unique())
selected_model = st.selectbox('Select Model', cars_data[cars_data['Manufacturer'] == manufacturer]['Model'].unique())
filtered_data = cars_data[(cars_data["Manufacturer"] == manufacturer) & (cars_data["Model"] == selected_model)]
id = st.number_input('ID (This value cannot be changed):', filtered_data["ID"].values[0], disabled=True)
#Son satır, cars_data DataFrame'ini seçilen üretici ve modele göre filtreler ve filtered_data adlı yeni bir DataFrame oluşturur.

#Streamlit kullanarak bir kullanıcı arayüzü oluşturalım ve çeşitli araç özelliklerinin seçilebilmesini sağlayalım. 
levy = st.number_input('Levy (This value cannot be changed):', filtered_data.iloc[0]["Levy"], disabled=True)
#levy = st.slider('Select Levy',int(cars_data['Levy'].min()), 2000)

prod_year = st.slider('Production Year', 1992, int(cars_data['Prod. year'].max()))
#Bu satır, kullanıcıya üretim yılını seçme olanağı sağlar. st.slider fonksiyonu, bir kaydırma çubuğu oluşturur ve minimum değeri 1992, maksimum değeri ise cars_data['Prod. year'].max() olarak ayarlar.

category = st.selectbox('Select Category', cars_data['Category'].unique())
#Bu satır, kullanıcıya cars_data DataFrame'inde bulunan benzersiz kategorilerden birini seçme olanağı sağlar.

leather_interior = st.selectbox('Leather Interior', ['Yes', 'No'])
#Bu satır, kullanıcıya deri iç döşeme seçeneğini seçme olanağı sağlar.

fuel_type = st.selectbox('Fuel Type', cars_data['Fuel type'].unique())
#Bu satır, kullanıcıya yakıt türlerinden birini seçme olanağı sağlar.

engine_volume = st.slider('Engine Volume (L)', 0.0, 10.0, step=0.1)
#Bu satır, kullanıcıya motor hacmini seçme olanağı sağlar.

mileage = st.slider('Mileage (km)', float(cars_data['Mileage'].min()), 1000000.0, step=1.0)
#Bu satır, kullanıcıya kilometreyi seçme olanağı sağlar.

cylinders = st.slider('Number of Cylinders', int(cars_data['Cylinders'].min()), int(cars_data['Cylinders'].max()))
#Bu satır, kullanıcıya silindir sayısını seçme olanağı sağlar.

gear_box_type = st.selectbox('Gear Box Type', cars_data['Gear box type'].unique())
#Bu satır, kullanıcıya vites kutusu türlerinden birini seçme olanağı sağlar.

drive_wheels = st.selectbox('Drive Wheels', cars_data['Drive wheels'].unique())
#Bu satır, kullanıcıya çekiş türlerinden birini seçme olanağı sağlar.

doors = st.selectbox('Number of Doors', cars_data['Doors'].unique())
#Bu satır, kullanıcıya kapı sayılarından birini seçme olanağı sağlar.

wheel = st.selectbox('Steering Wheel Position', cars_data['Wheel'].unique())
#Bu satır, kullanıcıya direksiyon pozisyonunu seçme olanağı sağlar.

color = st.selectbox('Color', cars_data['Color'].unique())
#Bu satır, kullanıcıya renklerden birini seçme olanağı sağlar.

airbags = st.slider('Number of Airbags', int(cars_data['Airbags'].min()), int(cars_data['Airbags'].max()))
#u satır, kullanıcıya hava yastığı sayısını seçme olanağı sağlar.

#Streamlit kullanarak bir tahmin butonu oluşturalım ve butona tıklandığında tahmin için gerekli olan giriş verilerini bir DataFrame'e eklesin
if st.button("Predict"):
    input_data = pd.DataFrame({
        'ID': [id],
        'Levy': [levy],
        'Manufacturer': [manufacturer],
        'Model': [selected_model],
        'Prod. year': [prod_year],
        'Category': [category],
        'Leather interior': [leather_interior],
        'Fuel type': [fuel_type],
        'Engine volume': [engine_volume],
        'Mileage': [mileage],
        'Cylinders': [cylinders],
        'Gear box type': [gear_box_type],
        'Drive wheels': [drive_wheels],
        'Doors': [doors],
        'Wheel': [wheel],
        'Color': [color],
        'Airbags': [airbags]
    })


    # Kategorik değişkenleri eğitim aşamasındaki gibi kodlayalım (verileri sayısal değerlere dönüştürdük)
    input_data['Model'].replace({
    'Getz': 401, 'Ghibli': 402, 'Golf': 403, 'Golf GTI': 404, 'Golf TDI': 405, 
    'Grand Cherokee': 406, 'Grandeur': 407, 'H1': 408, 'H1 GRAND STAREX': 409, 
    'H3': 410, 'HHR': 411, 'HS 250h': 412, 'HS 250h Hybrid': 413, 'HUSTLER': 414, 
    'Highlander': 415, 'Highlander 2,4': 416, 'Highlander 2.4 lit': 417, 
    'Highlander LIMITED': 418, 'Highlander sport': 419, 'Hilux': 420, 
    'Hr-v': 421, 'Hr-v EX': 422, 'Hr-v EXL': 423, 'I': 424, 'I30': 425, 
    'IS 200': 426, 'IS 250': 427, 'IS 250 TURBO': 428, 'IS 300': 429, 
    'IS 350': 430, 'ISIS': 431, 'Impala': 432, 'Impreza': 433, 'Insight': 434, 
    'Insight EX': 435, 'Ioniq': 436, 'Ipsum': 437, 'Ist': 438, 'JX35': 439, 
    'Jetta': 440, 'Jetta 1.4 TURBO': 441, 'Jetta Hybrid': 442, 'Jetta SE': 443, 
    'Jetta SPORT': 444, 'Jetta TDI': 445, 'Jetta s': 446, 'Jetta se': 447, 
    'Jetta sei': 448, 'Jetta sport': 449, 'Jetta სასწაფოდ': 450, 'Jetta სპორტ': 451, 
    'Jimny GLX': 452, 'Journey': 453, 'Juke': 454, 'Juke NISMO': 455, 
    'Juke Nismo': 456, 'Juke Nismo RS': 457, 'Kalos': 458, 'Kangoo': 459, 
    'Kicks': 460, 'Kicks SR': 461, 'Kizashi': 462, 'Korando': 463, 
    'Kyron': 464, 'LS 460': 465, 'LX 570': 466, 'Lacetti': 467, 'Lancer GT': 468, 
    'Land Cruiser': 469, 'Land Cruiser 80': 470, 'Land Cruiser Prado': 471, 
    'Land Rover Sport': 472, 'Lantra': 473, 'Lantra LIMITED': 474, 'Leaf': 475, 
    'Legacy': 476, 'Liberty': 477, 'M3': 478, 'M4': 479, 'M5': 480, 'M6': 481, 
    'MDX': 482, 'MKZ': 483, 'MKZ hybrid': 484, 'ML 250': 485, 'ML 280': 486, 
    'ML 350': 487, 'ML 350 4 MATIC': 488, 'ML 350 BLUETEC': 489, 
    'ML 350 sport': 490, 'ML 550': 491, 'Malibu': 492, 'Malibu Hybrid': 493, 
    'Malibu LT': 494, 'Malibu eco': 495, 'March': 496, 'Mariner': 497, 
    'Mariner Hybrid': 498, 'Matiz': 499, 'Maxima': 500, 'Mazda 2': 501, 
    'Mazda 3': 502, 'Mazda 3 1.6': 503, 'Mazda 3 2.0': 504, 'Mazda 3 2.3': 505, 
    'Mazda 6': 506, 'Mazda 6 2.0': 507, 'Mazda 6 2.3': 508, 'Mazda CX-3': 509, 
    'Mazda CX-5': 510, 'Mazda CX-5 AWD': 511, 'Mazda CX-9': 512, 
    'Mazda MX-5': 513, 'Mazda RX-8': 514, 'Mii': 515, 'Mitsubishi': 516, 
    'Mitsubishi ASX': 517, 'Mitsubishi Eclipse': 518, 'Mitsubishi Galant': 519, 
    'Mitsubishi L200': 520, 'Mitsubishi Lancer': 521, 'Mitsubishi Mirage': 522, 
    'Mitsubishi Outlander': 523, 'Mitsubishi Outlander Sport': 524, 
    'Mitsubishi Shogun': 525, 'Mitsubishi Space Star': 526, 'Murano': 527, 
    'Mustang': 528, 'Mustang GT': 529, 'Nautilus': 530, 'Navara': 531, 
    'Nissan': 532, 'Nissan Altima': 533, 'Nissan Armada': 534, 'Nissan Cube': 535, 
    'Nissan Juke': 536, 'Nissan Leaf': 537, 'Nissan Maxima': 538, 'Nissan Micra': 539, 
    'Nissan Murano': 540, 'Nissan Pathfinder': 541, 'Nissan Rogue': 542, 
    'Nissan Sentra': 543, 'Nissan Titan': 544, 'Nissan Versa': 545, 'Nissan X-Trail': 546, 
    'NX 200t': 547, 'NX 300': 548, 'NX 300h': 549, 'Outback': 550, 'Outlander': 551, 
    'Outlander PHEV': 552, 'Pajero': 553, 'Peugeot': 554, 'Peugeot 2008': 555, 
    'Peugeot 208': 556, 'Peugeot 3008': 557, 'Peugeot 308': 558, 'Peugeot 5008': 559, 
    'Peugeot 508': 560, 'Peugeot Boxer': 561, 'Peugeot Expert': 562, 
    'Peugeot Partner': 563, 'Peugeot Rifter': 564, 'Peugeot Traveller': 565, 
    'Polo': 566, 'Porsche': 567, 'Porsche 911': 568, 'Porsche Cayenne': 569, 
    'Porsche Macan': 570, 'Porsche Panamera': 571, 'Porsche Taycan': 572, 
    'Renault': 573, 'Renault Clio': 574, 'Renault Captur': 575, 'Renault Koleos': 576, 
    'Renault Laguna': 577, 'Renault Megane': 578, 'Renault Talisman': 579, 
    'Renault Trafic': 580, 'Renault Zoe': 581, 'Rover': 582, 'RX 300': 583, 
    'RX 450h': 584, 'RX 500h': 585, 'Scion': 586, 'Sienna': 587, 'Sonata': 588, 
    'Subaru': 589, 'Subaru Forester': 590, 'Subaru Impreza': 591, 'Subaru Outback': 592, 
    'Subaru WRX': 593, 'Suzuki': 594, 'Suzuki Alto': 595, 'Suzuki Baleno': 596, 
    'Suzuki Grand Vitara': 597, 'Suzuki Ignis': 598, 'Suzuki Jimny': 599, 
    'Suzuki Swift': 600, 'Suzuki Vitara': 601, 'Suzuki Wagon R': 602, 
    'Tesla': 603, 'Tesla Model 3': 604, 'Tesla Model S': 605, 'Tesla Model X': 606, 
    'Tesla Model Y': 607, 'TOYOTA': 608, 'Toyota Auris': 609, 'Toyota Avensis': 610, 
    'Toyota Aygo': 611, 'Toyota Camry': 612, 'Toyota Corolla': 613, 'Toyota FJ Cruiser': 614, 
    'Toyota Highlander': 615, 'Toyota Land Cruiser': 616, 'Toyota Prius': 617, 
    'Toyota RAV4': 618, 'Toyota Supra': 619, 'Toyota Tacoma': 620, 'Toyota Yaris': 621, 
    'V8': 622, 'Venza': 623, 'Vios': 624, 'Volkswagen': 625, 'Volkswagen Beetle': 626, 
    'Volkswagen Golf': 627, 'Volkswagen Jetta': 628, 'Volkswagen Passat': 629, 
    'Volkswagen Tiguan': 630, 'Volvo': 631, 'Volvo XC40': 632, 'Volvo XC60': 633, 
    'Volvo XC90': 634
    }, inplace=True)

    input_data['Manufacturer'].replace({
    'ACURA': 0, 'AUDI': 1, 'BENTLEY': 2, 'BMW': 3, 'BUICK': 4, 'CADILLAC': 5, 
    'CHEVROLET': 6, 'CHRYSLER': 7, 'CITROEN': 8, 'DAEWOO': 9, 'DAIHATSU': 10, 
    'DODGE': 11, 'FERRARI': 12, 'FIAT': 13, 'FORD': 14, 'GAZ': 15, 'GMC': 16, 
    'HONDA': 17, 'HUMMER': 18, 'HYUNDAI': 19, 'INFINITI': 20, 'JAGUAR': 21, 
    'JEEP': 22, 'KIA': 23, 'LAND ROVER': 24, 'LEXUS': 25, 'LINCOLN': 26, 
    'MASERATI': 27, 'MAZDA': 28, 'MERCEDES-BENZ': 29, 'MERCURY': 30, 
    'MINI': 31, 'MITSUBISHI': 32, 'NISSAN': 33, 'OPEL': 34, 'PEUGEOT': 35, 
    'PORSCHE': 36, 'RENAULT': 37, 'SCION': 38, 'SKODA': 39, 'SSANGYONG': 40, 
    'SUBARU': 41, 'SUZUKI': 42, 'TESLA': 43, 'TOYOTA': 44, 'VAZ': 45, 
    'VOLKSWAGEN': 46, 'VOLVO': 47
    }, inplace=True)

    input_data['Category'].replace({
    'Cabriolet': 0, 'Coupe': 1, 'Goods wagon': 2, 'Hatchback': 3, 'Jeep': 4, 
    'Microbus': 5,  'Minivan': 6, 'Pickup': 7, 'Sedan': 8, 'Universal': 9
    }, inplace=True)

    input_data['Leather interior'].replace({'Yes': 1, 'No': 0}, inplace=True)

    input_data['Fuel type'].replace({
        'CNG': 0, 'Diesel': 1, 'Hybrid': 2, 'LPG': 3, 'Petrol': 4, 'Plug-in Hybrid': 5, 'Hydrogen': 6
    }, inplace=True)

    input_data['Gear box type'].replace({
        'Automatic': 0, 'Manual': 1, 'Tiptronic': 2, 'Variator': 3
    }, inplace=True)

    input_data['Drive wheels'].replace({
        '4x4': 0, 'Front': 1, 'Rear': 2
    }, inplace=True)

    input_data['Doors'].replace({
        '02-Mar': 0, '04-May': 1, '>5': 2
    }, inplace=True)

    input_data['Wheel'].replace({
        'Left wheel': 0, 'Right-hand drive': 1
    }, inplace=True)

    input_data['Color'].replace({
        'Beige': 0, 'Black': 1, 'Blue': 2, 'Brown': 3, 'Carnelian red': 4, 'Golden': 5, 
        'Green': 6, 'Grey': 7, 'Orange': 8, 'Pink': 9, 'Purple': 10, 'Red': 11, 
        'Silver': 12, 'Sky blue': 13, 'White': 14, 'Yellow': 15
    }, inplace=True)


    #cars_data DataFrame'inde bulunan benzersiz üretici ve model isimlerini sayısal değerlere dönüştürelim
    manufacturer_mapping = {name: idx for idx, name in enumerate(cars_data['Manufacturer'].unique())}
    model_mapping = {name: idx for idx, name in enumerate(cars_data['Model'].unique())}
    input_data['Manufacturer'] = input_data['Manufacturer'].map(manufacturer_mapping)
    input_data['Model'] = input_data['Model'].map(model_mapping)
        #unique benzersiz değerleri döndürür. enumerate fonksiyonu, bu benzersiz değerler listesini alır ve her bir değere bir indeks atar.
        #map fonksiyonu, manufacturer_mapping sözlüğündeki anahtar-değer çiftlerini kullanarak sayısal değerlere dönüşümü gerçekleştirir.


    try:
        car_price = model.predict(input_data)  #Bu satır, model değişkenindeki eğitilmiş model kullanılarak input_data verileri üzerinde tahmin yapar ve sonucu car_price değişkenine atar.
        st.markdown(f'### Predicted Car Price: ${car_price[0]:,.2f}')  #Bu satır, tahmin edilen araç fiyatını Streamlit kullanarak ekrana yazdırır. 
        #f-string kullanarak tahmin edilen fiyat dinamik olarak eklenir ve iki ondalık basamakla formatlanır.

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

    #Bu kod satırları, model.predict(input_data) fonksiyonunu kullanarak input_data verileri üzerinde tahmin yapmaya çalışır 
    #tahmin sonucunu Streamlit kullanarak ekrana yazdırır. Eğer bir hata oluşursa, hata mesajını ekrana yazdırır    