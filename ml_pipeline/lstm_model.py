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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data_models import AttackScenario


class LstmDosDetectionPipeline:
    def __init__(self, dataset_path: str = "./dataset2.csv", attackers_path: str = "./container_results_2.json",
                 sequence_length: int = 60):
        print(f"[*] Inicjalizacja potoku Głębokie Uczenia LSTM (Horyzont czasowy = {sequence_length} pakietów)...")
        self.df: DataFrame = pd.read_csv(dataset_path)
        self.attack_scenario: AttackScenario = AttackScenario.from_json(attackers_path)

        self.sequence_length = sequence_length
        self.label_encoder = LabelEncoder()

        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.model = None
        self.num_classes = 0

    def label_data(self):
        print("\n[*] Etykietowanie danych wieloklasowych (Multiclass)...")
        self.df['label'] = 'normal'

        dt_timestamps = pd.to_datetime(self.df['ts'], format='mixed', errors='coerce')
        event_timestamps = dt_timestamps.astype('int64') / 10 ** 9

        for attacker in self.attack_scenario.attackers:
            print(f"  -> Aplikowanie etykiety: {attacker.attack_type} dla IP: {attacker.ip}")
            ip_match_condition = (self.df['id.orig_h'] == attacker.ip) | (self.df['id.resp_h'] == attacker.ip)
            time_condition = (event_timestamps >= attacker.start_ts) & (event_timestamps <= attacker.end_ts)
            final_condition = ip_match_condition & time_condition
            self.df.loc[final_condition, 'label'] = attacker.attack_type

        print(f"[+] Rozkład klas w zbiorze surowym:\n{self.df['label'].value_counts()}")

    def clean_dataset(self, null_threshold=0.95):
        print(f"\n[*] Czyszczenie datasetu (usuwanie kolumn z >{null_threshold * 100}% braków)...")
        null_rates = self.df.isnull().mean()
        columns_to_drop = null_rates[null_rates > null_threshold].index.tolist()

        if columns_to_drop:
            print(f"  -> Usunięto {len(columns_to_drop)} pustych kolumn.")
            self.df = self.df.drop(columns=columns_to_drop)
        print(f"  -> Pozostało kolumn: {len(self.df.columns)}")

    def prepare_data_3d(self):
        print("\n[*] Przygotowanie Tensorów 3D dla sieci LSTM...")

        # 1. Sortowanie chronologiczne
        self.df['ts_datetime'] = pd.to_datetime(self.df['ts'], format='mixed', errors='coerce')
        self.df = self.df.sort_values(by='ts_datetime').reset_index(drop=True)

        # 2. Wybór czystych cech bazowych (LSTM samo wyciągnie z nich czas)
        keep_numeric = [
            'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts',
            'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
            'request_body_len', 'response_body_len'
        ]
        existing_numeric = [col for col in keep_numeric if col in self.df.columns]
        X_raw = self.df[existing_numeric].copy()

        # Konwersja formatów i wypełnianie NaN
        for col in existing_numeric:
            if X_raw[col].dtype == 'object':
                X_raw[col] = pd.to_timedelta(X_raw[col], errors='coerce').dt.total_seconds()
            X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce').fillna(0.0).astype(float)

        print("  -> Skalowanie cech (StandardScaler)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw.to_numpy())

        # Kodowanie etykiet tekstowych -> One-Hot
        y_encoded = self.label_encoder.fit_transform(self.df['label'].to_numpy())
        self.num_classes = len(self.label_encoder.classes_)

        # 3. Budowanie sekwencji
        print(f"  -> Generowanie okien przesuwnych (Długość: {self.sequence_length})...")
        X_3d, y_3d = [], []

        for i in range(len(X_scaled) - self.sequence_length):
            X_3d.append(X_scaled[i: i + self.sequence_length])
            y_3d.append(y_encoded[i + self.sequence_length - 1])

        X_3d = np.array(X_3d, dtype=np.float32)
        y_3d = np.array(y_3d, dtype=np.int32)
        y_one_hot = to_categorical(y_3d, num_classes=self.num_classes)

        # 4. Podział na zbiór treningowy i testowy
        print("[*] Podział na zbiór treningowy i testowy (80/20)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_3d, y_one_hot, test_size=0.2, random_state=42, stratify=y_3d
        )
        print(f"[+] Macierz treningowa: {self.X_train.shape}")

    def build_lstm(self):
        print("\n[*] Budowanie architektury sieci rekurencyjnej LSTM...")
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])

        self.model = Sequential([
            # Pierwsza warstwa LSTM z pamięcią wsteczną
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),

            # Druga warstwa LSTM (skupiająca wyekstrahowane cechy w jeden wektor)
            LSTM(32, return_sequences=False),
            Dropout(0.3),

            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train_model(self, epochs: int = 100, batch_size: int = 128):
        print(f"\n[*] Rozpoczęcie uczenia LSTM (Max epok: {epochs}, Batch: {batch_size})...")

        # Obliczenie wag klas, by zapobiec ignorowaniu ataków przez sieć
        y_integers = np.argmax(self.y_train, axis=1)
        classes, counts = np.unique(y_integers, return_counts=True)
        class_weights = {cls: len(y_integers) / (len(classes) * count) for cls, count in zip(classes, counts)}
        print("  -> Zastosowano zbalansowane wagi klas.")

        # Mechanizm ucinania treningu w idealnym momencie
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            class_weight=class_weights,
            callbacks=[early_stopping],
            verbose=1
        )
        print("[+] Model LSTM pomyślnie wytrenowany!")
        return history

    def evaluate_model(self):
        print("\n=== EWALUACJA MODELU LSTM ===")

        y_pred_probs = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        class_names = self.label_encoder.classes_

        print("\nRaport Klasyfikacji (LSTM Deep Learning):")
        print(classification_report(y_true, y_pred, target_names=class_names))

        print("\nGenerowanie wieloklasowej macierzy pomyłek...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    # Używam zielonego dla wizualnego odróżnienia od RF(Niebieski) i CNN(Pomarańcz)
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.ylabel('Rzeczywista klasa')
        plt.xlabel('Przewidziana klasa')
        plt.title('Macierz Błędów - Detekcja LSTM (Stealthy DoS)')
        plt.tight_layout()
        plt.savefig("lstm_confusion_matrix.png")
        print("[+] Wykres zapisano do lstm_confusion_matrix.png")
        plt.show()


def main():
    # Uruchamiamy potok. Długie okno (60) dla ataków stealthy to absolutna konieczność!
    pipeline = LstmDosDetectionPipeline(sequence_length=100)

    pipeline.label_data()
    pipeline.clean_dataset(null_threshold=0.95)
    pipeline.prepare_data_3d()

    pipeline.build_lstm()
    # Puszczamy na 100 epok, ale EarlyStopping i tak utnie to w odpowiednim momencie (pewnie ok. 50-80 epoki)
    pipeline.train_model(epochs=100, batch_size=128)
    pipeline.evaluate_model()


if __name__ == "__main__":
    main()