import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM

from data_models import AttackScenario


class CnnDosDetectionPipeline:
    def __init__(self, dataset_path: str = "./dataset2.csv", attackers_path: str = "./container_results_2.json",
                 sequence_length: int = 15):
        print("[*] Inicjalizacja potoku głębokiego uczenia 1D-CNN...")
        self.df: DataFrame = pd.read_csv(dataset_path)
        self.attack_scenario: AttackScenario = AttackScenario.from_json(attackers_path)

        # Parametr określający, jak długie sekwencje czasowe analizuje sieć (np. okno ostatnich 15 logów sieciowych)
        self.sequence_length = sequence_length

        # Słowniki i koder etykiet dla Keras
        self.label_encoder = LabelEncoder()

        # Macierze podziału danych 3D
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.model = None

    def label_data(self):
        print("\n[*] Etykietowanie danych...")
        self.df['label'] = 'normal'

        dt_timestamps = pd.to_datetime(self.df['ts'], format='mixed', errors='coerce')
        event_timestamps = dt_timestamps.astype('int64') / 10 ** 9

        for attacker in self.attack_scenario.attackers:
            print(f"  -> Aplikowanie etykiety wieloklasowej: {attacker.attack_type} dla IP: {attacker.ip}")
            ip_match_condition = (self.df['id.orig_h'] == attacker.ip) | (self.df['id.resp_h'] == attacker.ip)
            time_condition = (event_timestamps >= attacker.start_ts) & (event_timestamps <= attacker.end_ts)
            final_condition = ip_match_condition & time_condition
            self.df.loc[final_condition, 'label'] = attacker.attack_type

        print(f"[+] Rozkład klas w zbiorze surowym:\n{self.df['label'].value_counts()}")

    def clean_dataset(self, null_threshold=0.5):
        """
        Usuwa kolumny, w których odsetek brakujących wartości (NaN/Null)
        przekracza podany próg (domyślnie 50%).
        """
        print(f"\n[*] Czyszczenie datasetu: Usuwanie kolumn z brakami danych (próg > {null_threshold * 100}%)...")

        # 1. Obliczanie odsetka braków dla każdej kolumny
        null_rates = self.df.isnull().mean()

        # 2. Wybieranie nazw kolumn, które przekraczają próg
        columns_to_drop = null_rates[null_rates > null_threshold].index.tolist()

        # 3. Usuwanie i raportowanie
        if columns_to_drop:
            print(f"  -> Usunięto {len(columns_to_drop)} kolumn:")
            for col in columns_to_drop:
                print(f"     - {col} ({null_rates[col] * 100:.1f}% nulls)")

            self.df = self.df.drop(columns=columns_to_drop)
        else:
            print("  -> Nie usunięto żadnych kolumn. Wszystkie spełniają kryteria.")

        print(f"  -> Pozostało kolumn w datasecie: {len(self.df.columns)}")

    def prepare_data_3d(self):
        print("\n[*] Zaawansowane przygotowanie danych dla 1D-CNN (Tworzenie Tensorów 3D)...")

        # 1. Sortowanie chronologiczne - fundamentalne dla prawidłowego działania sieci CNN w czasie
        self.df['ts_datetime'] = pd.to_datetime(self.df['ts'], format='mixed', errors='coerce')
        self.df = self.df.sort_values(by='ts_datetime').reset_index(drop=True)

        # 2. Selekcja bazowych, surowych cech numerycznych protokołu (Bez ręcznych rolling windows!)
        keep_numeric = [
            'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts',
            'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
            'request_body_len', 'response_body_len'
        ]

        existing_numeric = [col for col in keep_numeric if col in self.df.columns]

        # Wyciągamy dane numeryczne i uzupełniamy braki
        X_raw = self.df[existing_numeric].copy()
        for col in existing_numeric:
            if X_raw[col].dtype == 'object':
                X_raw[col] = pd.to_timedelta(X_raw[col], errors='coerce').dt.total_seconds()
            X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce').fillna(0.0).astype(float)

        # Sieci neuronowe bezwzględnie wymagają standaryzacji (skalowania) danych wejściowych
        print("  -> Skalowanie cech wejściowych (StandardScaler)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw.to_numpy())

        # Kodowanie etykiet tekstowych na numeryczne (0, 1, 2...)
        y_encoded = self.label_encoder.fit_transform(self.df['label'].to_numpy())
        self.num_classes = len(self.label_encoder.classes_)

        # 3. Budowanie sekwencji 3D kroczących dla filtrów splotowych
        # Kształt tensora wejściowego: (liczba_próbek, długość_sekwencji, liczba_cech)
        print(f"  -> Generowanie okien przesuwnych o długości sekwencji: {self.sequence_length}")
        X_3d, y_3d = [], []

        for i in range(len(X_scaled) - self.sequence_length):
            X_3d.append(X_scaled[i: i + self.sequence_length])
            # Etykietą sekwencji jest stan ostatniego zdarzenia sieciowego w oknie
            y_3d.append(y_encoded[i + self.sequence_length - 1])

        X_3d = np.array(X_3d, dtype=np.float32)
        y_3d = np.array(y_3d, dtype=np.int32)

        # Konwersja etykiet na format One-Hot (wymóg dla kategorycznej entropii krzyżowej w Keras)
        y_one_hot = to_categorical(y_3d, num_classes=self.num_classes)

        # 4. Podział na zbiór treningowy i testowy
        print("[*] Podział tensorów 3D na zbiór treningowy i testowy (80/20)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_3d, y_one_hot, test_size=0.2, random_state=42, stratify=y_3d
        )
        print(f"[+] Dane przygotowane. Kształt macierzy treningowej: {self.X_train.shape}")

    def build_cnn(self):
        print("\n[*] Budowanie architektury głębokiej sieci splotowej 1D-CNN...")
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])  # (długość_sekwencji, liczba_cech)

        self.model = Sequential([
            # Pierwsza warstwa splotowa - uczy się lokalnych mikro-wzorców czasowych
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            # Druga warstwa splotowa - uczy się bardziej złożonych relacji makroskopowych
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),

            # Spłaszczenie danych do warstw klasyfikacyjnych
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.4),
            # Warstwa wyjściowa z funkcją Softmax dla klasyfikacji wieloklasowej
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()



    def build_lstm(self):
        print("\n[*] Budowanie architektury sieci LSTM...")
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])

        self.model = Sequential([
            # Zamiast Conv1D wstawiamy warstwę LSTM z pamięcią wsteczną
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),

            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train_model(self, epochs: int = 100, batch_size: int = 128):
        print(f"\n[*] Uruchomienie uczenia sieci neuronowej (Maksymalnie Epochs={epochs}, Batch Size={batch_size})...")

        y_integers = np.argmax(self.y_train, axis=1)
        classes, counts = np.unique(y_integers, return_counts=True)
        total_samples = len(y_integers)
        class_weights = {cls: total_samples / (len(classes) * count) for cls, count in zip(classes, counts)}

        # EMERYTURA DLA MODELU (Early Stopping)
        # Monitoruje val_loss. Jeśli nie spadnie przez 7 epok z rzędu (patience), przerywa trening.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            class_weight=class_weights,
            callbacks=[early_stopping],  # <--- DODAJEMY CALLBACK
            verbose=1
        )
        print("[+] Sieć CNN została pomyślnie wytrenowana!")
        return history

    def evaluate_model(self):
        print("\n=== EWALUACJA MODELU 1D-CNN ===")

        # Pobieramy predykcje prawdopodobieństw z sieci i wyciągamy indeksy o najwyższej wartości
        y_pred_probs = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Odwracamy kodowanie One-Hot dla zbioru testowego, aby uzyskać oryginalne indeksy klas
        y_true = np.argmax(self.y_test, axis=1)

        # Mapujemy numery klas z powrotem na czytelne nazwy tekstowe (normal, slowloris, etc.)
        class_names = self.label_encoder.classes_

        print("\nRaport Klasyfikacji (Deep Learning):")
        print(classification_report(y_true, y_pred, target_names=class_names))

        print("\nGenerowanie wieloklasowej macierzy pomyłek dla sieci CNN...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.ylabel('Rzeczywista klasa')
        plt.xlabel('Przewidziana klasa')
        plt.title('Macierz Błędów - Detekcja 1D-CNN DoS')
        plt.tight_layout()
        plt.savefig("cnn_confusion_matrix.png")
        print("[+] Wykres macierzy pomyłek zapisano do cnn_confusion_matrix.png")
        plt.show()


def main():
    # Definiujemy sekwencję (okno wejściowe o długości 15 pakietów pod rząd)
    pipeline = CnnDosDetectionPipeline(sequence_length=100)

    pipeline.label_data()
    pipeline.clean_dataset(null_threshold=0.95)
    pipeline.prepare_data_3d()

    pipeline.build_cnn()
    pipeline.train_model(epochs=100, batch_size=128)
    pipeline.evaluate_model()


if __name__ == "__main__":
    main()