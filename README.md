# Turkish

---

# **Balık Türleri Sınıflandırması için Yapay Sinir Ağları (ANN)**

### **Proje Genel Bakışı**
Bu projenin amacı, balık türlerini görüntüleri kullanarak sınıflandırmaktır. Bu amaçla, "A Large Scale Fish Dataset" (Büyük Ölçekli Balık Veri Seti) üzerinde eğitilen bir Yapay Sinir Ağı (ANN) modeli geliştirilmiştir. Proje, veri ön işleme, veri artırma, model eğitimi ve değerlendirmesi gibi birçok aşamayı içermektedir. Ana hedef, görüntü verilerini kullanarak balık türlerini doğru bir şekilde sınıflandırabilecek bir derin öğrenme modeli oluşturmaktır.

---

## **Kullanılan Teknolojiler**
Proje boyunca veri işleme, model oluşturma ve değerlendirme süreçlerini kolaylaştırmak için çeşitli araçlar ve teknolojiler kullanılmıştır. Bunlar:

- **Python**: Veri işleme, model oluşturma ve değerlendirme dahil olmak üzere tüm iş akışının uygulanmasında kullanılan ana programlama dili.
- **TensorFlow ve Keras**: ANN modelini oluşturmak ve eğitmek için kullanılmıştır. Keras, TensorFlow üzerine inşa edilmiş olup, model mimarisinin oluşturulmasını ve eğitimi kolaylaştıran fonksiyonlar sağlamıştır.
- **Pandas ve NumPy**: Veri manipülasyonu ve sayısal işlemler için kullanılmıştır, veri setinin verimli bir şekilde işlenmesine olanak tanımıştır.
- **Matplotlib ve Seaborn**: Veri seti ve model performansını görselleştirmek için kullanılmıştır. Eğitim süresince doğruluk ve kayıp gibi metriklerin görselleştirilmesi sağlanmıştır.
- **Scikit-learn**: Veri setinin eğitim, doğrulama ve test setlerine bölünmesi ve doğruluk, kesinlik, geri çağırma ve F1 skoru gibi çeşitli değerlendirme metriklerinin hesaplanması için kullanılmıştır.
- **Kaggle API**: Balık veri setini sorunsuz bir şekilde indirmek için kullanılmıştır.
- **Pillow (PIL)**: Veri hazırlama aşamasında görüntü işleme ve dönüştürme işlemlerine yardımcı olmuştur.

---

## **Proje İş Akışı**

### 1. **Veri Setinin Temin Edilmesi**
Veri seti, Kaggle API aracılığıyla sağlanmış olup, büyük ölçekli bir balık veri seti kullanılmıştır. Veri seti, her bir balık türü için ayrı klasörler halinde düzenlenmiştir.

### 2. **Veri Ön İşleme**
Modelin eğitilmesinden önce veri seti ön işleme tabi tutulmuştur. Görüntü yolları ve ilgili etiketler çıkarılmış ve veri seti eğitim, doğrulama ve test setlerine bölünmüştür. Bu adım, tüm alt setlerde balık türlerinin dengeli bir şekilde dağıtılmasını sağlamış ve modelin genelleme yeteneğini artırmıştır.

### 3. **Veri Artırma**
Her tür için sınırlı sayıda görüntü bulunduğundan, veri artırma teknikleri uygulanarak veri seti yapay olarak genişletilmiştir. Bu teknikler arasında döndürme, yakınlaştırma, kaydırma ve yatay çevirme gibi işlemler yer almıştır. Veri artırma, eğitim sırasında modelin daha geniş bir görüntü çeşitliliğine maruz kalmasını sağlayarak aşırı uyum (overfitting) riskini azaltmıştır.

### 4. **Model Mimarisi**
Bu sınıflandırma görevi için Yapay Sinir Ağı (ANN) modeli tercih edilmiştir. Modelin mimarisi şu bileşenlerden oluşmaktadır:

- **Girdi Katmanı**: Görüntüler, sinir ağı ile uyumluluğu sağlamak için standart bir boyuta yeniden boyutlandırılmıştır.
- **Gizli Katmanlar**: Aktivasyon fonksiyonlarıyla donatılmış birden fazla tam bağlantılı (dense) katman kullanılmıştır. Bu katmanlar, verideki karmaşık desenleri öğrenmekle sorumludur.
- **Dropout Katmanları**: Aşırı uyumun önüne geçmek amacıyla eğitim sırasında belirli nöronların rastgele devre dışı bırakıldığı dropout uygulanmıştır.
- **Batch Normalization**: Öğrenme sürecini stabilize etmek için aktivasyonlar normalleştirilmiş, böylece modelin daha hızlı yakınsaması sağlanmıştır.
- **Çıkış Katmanı**: Son katman, balık türlerinin olasılıklarını çıktı olarak üretmek için softmax aktivasyon fonksiyonunu kullanmış ve çok sınıflı sınıflandırmayı mümkün kılmıştır.

### 5. **Modelin Derlenmesi**
Öğrenme sürecini optimize etmek için Adam optimizasyon algoritması tercih edilmiştir, çünkü bu algoritma büyük veri setleri ve yüksek boyutlu uzaylarda verimli çalışmaktadır. Kaybı ölçmek için çok sınıflı sınıflandırma görevlerine uygun olan kategorik çapraz entropi kullanılmıştır.

### 6. **Modelin Eğitilmesi**
Model, veri artırma ile genişletilen veri seti üzerinde birkaç dönem boyunca eğitilmiştir. Aşırı uyumu önlemek ve modelin verimli bir şekilde yakınsamasını sağlamak amacıyla erken durdurma (early stopping) ve öğrenme oranı düşürme stratejileri uygulanmıştır. Erken durdurma, modelin doğrulama setindeki performansını izleyerek, iyileşme durduğunda eğitimi durdurur. Öğrenme oranı düşürme ise doğrulama performansına bağlı olarak öğrenme oranını dinamik bir şekilde ayarlar.

### 7. **Modelin Değerlendirilmesi**
Eğitimden sonra model, test seti üzerinde çeşitli metrikler kullanılarak değerlendirilmiştir:

- **Doğruluk**: Doğru tahminlerin toplam tahminlere oranı.
- **F1 Skoru**: Kesinlik ve geri çağırma arasında bir denge sağlayan bu metrik, dengesiz veri setleriyle çalışırken oldukça faydalıdır.
- **Kesinlik ve Geri Çağırma**: Kesinlik, doğru pozitif tahminlerin doğruluğunu ölçerken, geri çağırma modelin tüm ilgili örnekleri bulma yeteneğini ölçer.
- **Karmaşıklık Matrisi**: Modelin hatalı tahminlerde bulunduğu kategorileri analiz ederek, yanlış sınıflandırma örüntülerini belirlememize yardımcı olmuştur.

### 8. **Sonuçların Görselleştirilmesi**
Eğitim süreci ve değerlendirme metrikleri, grafikler ve çizimler aracılığıyla görselleştirilmiştir. Eğitim ve doğrulama setlerindeki doğruluk ve kayıp gibi önemli metrikler dönemler boyunca izlenmiştir. Ayrıca karmaşıklık matrisi ve sınıflandırma raporları, model performansına ilişkin ayrıntılı bilgiler sağlamıştır.

### 9. **Hata Analizi**
Değerlendirme aşamasında modelin hatalı tahminlerde bulunduğu örnekler incelenmiştir. Bu analiz, iyileştirme potansiyeli olan alanları belirlemeye yardımcı olmuş ve model mimarisi veya veri hazırlama süreçlerinde yapılacak gelecekteki iyileştirmelere rehberlik etmiştir.

---

## **Sonuç**
Bu projede, görüntü verilerini kullanarak balık türlerini doğru bir şekilde sınıflandırabilen bir Yapay Sinir Ağı başarıyla geliştirilmiştir. Veri artırma, batch normalization ve dropout gibi teknikler kullanılarak modelin genel veri üzerindeki performansı iyileştirilmiştir. Değerlendirme süreci, modelin güçlü bir performans sergilediğini göstermiş, ancak gelecekte yapılabilecek iyileştirmeler için fırsatlar sunulmuştur.

---

## **Gelecek Çalışmalar**
Proje için önerilen bazı geliştirme ve iyileştirme adımları şunlardır:

- **Daha Gelişmiş Modellerin Keşfedilmesi**: ANN iyi bir performans sergilese de, görüntü tabanlı görevler için daha gelişmiş mimariler, örneğin Konvolüsyonel Sinir Ağları (CNN) denenebilir.
- **Transfer Öğrenme**: ResNet veya VGG16 gibi önceden eğitilmiş modellerin kullanılması, daha büyük veri setlerinden öğrenilen özelliklerin bu modelde kullanılmasını sağlayarak performansı artırabilir.
- **Veri Setinin Genişletilmesi**: Her balık türü için daha fazla görüntü eklemek veya yeni türler eklemek, modelin daha sağlam ve doğru hale gelmesine yardımcı olabilir.
- **Hiperparametre Optimizasyonu**: İlk hiperparametre optimizasyonları yapılmış olsa da, daha geniş bir parametre aralığını keşfetmek, modelin performansını daha da iyileştirebilir.

---

## **Kullanım Talimatları**
Bu projeyi yeniden üretmek için aşağıdaki adımları izleyin:
1. Depoyu GitHub'dan klonlayın.
2. Balık veri setini indirin ve bu yapılandırmaya göre entegre edin.
3. Tüm hücreleri çalıştırın.
4. Eğitim tamamlandığında, modelin performansını değerlendirmek 'Model Evaluation' altındaki hücreleri çalıştırabilirsiniz.

--- 


# Engish

---

# **Fish Species Classification using Artificial Neural Networks (ANN)**

### **Project Overview**
This project aims to classify various species of fish using images as input data. To achieve this, an Artificial Neural Network (ANN) was developed and trained on "A Large Scale Fish Dataset." The project involved multiple steps, including data preprocessing, augmentation, model training, and evaluation. The primary objective was to build a deep learning model that could accurately classify fish species based on image data.

---

## **Technologies Used**
Throughout the project, various tools and technologies were employed to facilitate data processing, model building, and evaluation. These include:

- **Python**: The primary language for implementing the entire workflow, including data manipulation, model building, and evaluation.
- **TensorFlow and Keras**: These libraries were used to construct and train the ANN model. Keras, built on TensorFlow, simplified the creation of the model architecture and provided functions for training and optimization.
- **Pandas and NumPy**: Essential libraries for data manipulation and numerical operations, enabling efficient handling of the dataset.
- **Matplotlib and Seaborn**: Used to visualize the dataset and model performance, including training accuracy and loss over time.
- **Scikit-learn**: Employed for splitting the dataset into training, validation, and test sets, and for computing various evaluation metrics such as accuracy, precision, recall, and F1 score.
- **Kaggle API**: Utilized to download the fish dataset efficiently.
- **Pillow (PIL)**: Assisted with image processing and conversion tasks during the data preparation stage.

---

## **Project Workflow**

### 1. **Dataset Acquisition**
The dataset was sourced from Kaggle using the API, allowing seamless integration of a large-scale fish dataset. This dataset contains multiple species of fish, with each species organized into separate folders.

### 2. **Data Preprocessing**
Before training the model, it was essential to preprocess the dataset. Image paths and corresponding labels were extracted, and the dataset was organized into training, validation, and test sets. This ensured a balanced distribution of fish species across all subsets, which is critical for the model to generalize well.

### 3. **Data Augmentation**
Given the relatively limited number of images per species, data augmentation techniques were applied to artificially expand the dataset. These techniques included rotation, zoom, shear, and horizontal flips, among others. Data augmentation helps the model generalize better by exposing it to a wider variety of images during training, mitigating the risk of overfitting.

### 4. **Model Architecture**
An Artificial Neural Network (ANN) was selected as the primary model architecture for this classification task. The model consisted of multiple layers, including:

- **Input Layer**: Images were resized to a standardized size to ensure compatibility with the neural network.
- **Hidden Layers**: Several fully connected (dense) layers with activation functions were employed. These layers are responsible for learning complex patterns within the data.
- **Dropout Layers**: Dropout was introduced to prevent overfitting by randomly disabling a portion of neurons during training.
- **Batch Normalization**: Batch normalization was used to stabilize the learning process, allowing the model to converge more quickly by normalizing activations.
- **Output Layer**: The final layer used a softmax activation function to output probabilities for each fish species, enabling multi-class classification.

### 5. **Model Compilation**
To optimize the learning process, the Adam optimizer was chosen due to its efficiency in handling large datasets and high-dimensional spaces. The loss function used was categorical cross-entropy, which is well-suited for multi-class classification problems.

### 6. **Model Training**
The model was trained using the augmented data over several epochs, with early stopping and learning rate reduction strategies employed to prevent overfitting and ensure the model converged efficiently. Early stopping monitors the model’s performance on the validation set and halts training when performance ceases to improve, while learning rate reduction adjusts the learning rate dynamically based on validation performance.

### 7. **Model Evaluation**
After training, the model's performance was evaluated on the test set using various metrics:

- **Accuracy**: A straightforward measure of the proportion of correct predictions.
- **F1 Score**: This metric provides a balance between precision and recall, making it particularly useful when dealing with imbalanced datasets.
- **Precision and Recall**: Precision measures the accuracy of positive predictions, while recall measures the model’s ability to identify all relevant instances.
- **Confusion Matrix**: This provided insight into the specific categories where the model made errors, helping identify patterns in misclassification.

### 8. **Results Visualization**
The training process and evaluation metrics were visualized using graphs and plots. Key metrics, such as accuracy and loss, were tracked across both training and validation sets over time. Additionally, confusion matrices and classification reports provided detailed insights into model performance across different fish species.

### 9. **Error Analysis**
An important aspect of the evaluation phase involved reviewing instances where the model made incorrect predictions. This analysis helped identify areas for potential improvement and guided future enhancements in model architecture or data preparation.

---

## **Conclusion**
The project successfully developed an Artificial Neural Network capable of classifying fish species with high accuracy. Through techniques such as data augmentation, batch normalization, and dropout, the model’s ability to generalize to unseen data was improved. The evaluation demonstrated strong performance, although there are opportunities for further enhancement.

---

## **Future Work**
Potential improvements and next steps for the project include:

- **Exploring More Advanced Models**: While the ANN performed well, experimenting with more sophisticated architectures, such as Convolutional Neural Networks (CNNs), may yield even better performance for image-based tasks.
- **Transfer Learning**: Utilizing pre-trained models such as ResNet or VGG16 may enhance model performance by leveraging features learned from much larger datasets.
- **Expanding the Dataset**: Increasing the number of images per fish species or incorporating new species could further improve the model’s robustness and accuracy.
- **Further Hyperparameter Tuning**: While initial hyperparameter tuning was performed, exploring a wider range of parameters could optimize the model further.

---

## **Instructions for Use**
To reproduce this project, follow these steps:
1. Clone the repository from GitHub.
2. Download the fish dataset and integrate it following the structure described in the repository.
3. Run the provided Jupyter notebook to train, evaluate, and visualize the model.

---
