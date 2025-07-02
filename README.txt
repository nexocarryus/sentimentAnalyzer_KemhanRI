# LIBRARY YANG HARUS DI INSTALL DAN DISIAPKAN

-install pandas
-install flask
-install glob
-install os
-install nltk
-install scikit-learn
-install joblib
-install Sastrawi
-install re

# PENJELASAN DIRECTORY

- models	: berfungsi untuk menyimpan dan melatih model machine learning
- static 	: berfungsi untuk menyimpan file css atau gambar 
- templates 	: berfungsi untuk menyimpan file frontend web 
- uploads 	: berfungsi untuk menyimpan file yang di upload oleh user dan menyimpan file akhir hasil prediksi

# TUTORIAL MENJALANKAN PROGRAM


-siapkan dataset untuk melatih program, pastikan dataset tersebut memiliki kolom 'mentions' dan kolom 'sentiment'
-jalankan trainModel.py untuk melatih dan menghasilkan model
-jalankan app.py untuk menjalankan aplikasi Utama
-masukan dataset dengan format csv yang ingin di analisis sentimennya
-tunggu hingga model selesai melakukan analisis sentiment
-tekan button download untuk mendapatkan hasil akhir dataset yang sudah berisi sentiment.



