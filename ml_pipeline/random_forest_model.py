import pandas as pd
import numpy as np
from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_models import AttackScenario

class DosDetectionPipeline:
    def __init__(self, dataset_path: str = "./dataset2.csv", attackers_path: str = "./container_results_2.json"):
        self.df: DataFrame = pd.read_csv(dataset_path)
        self.attack_scenario: AttackScenario = AttackScenario.from_json(attackers_path)
        self.model = RandomForestClassifier(
            n_estimators=150,  # Większa liczba drzew ustabilizuje proces decyzyjny
            max_depth=12,  # Ograniczenie głębokości zapobiega overfittingowi do wzorca Locusta
            min_samples_split=5,  # Drzewo nie stworzy gałęzi dla pojedynczego, losowego pakietu
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Użyj wszystkich rdzeni procesora do przyspieszenia treningu
        )
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def print_data_info(self):
        self.df.info()

    def label_data(self):
        print("\n[*] Labeling data...")

        # Set default values
        self.df['label'] = 'normal'
        self.df['binary_label'] = 0

        # Przygotowanie timestampów do precyzyjnego filtrowania
        # Bezwarunkowa konwersja na format datetime, a następnie na uniksowy timestamp (float w sekundach)
        # Używamy errors='coerce', aby zignorować ewentualne uszkodzone wiersze
        dt_timestamps = pd.to_datetime(self.df['ts'], format='mixed', errors='coerce')
        event_timestamps = dt_timestamps.astype('int64') / 10 ** 9

        # 2. Pętla po naszych obiektach Attacker (z wczytanego JSONa)
        for attacker in self.attack_scenario.attackers:
            print(f"  -> Aplikowanie etykiety: {attacker.attack_type} dla IP: {attacker.ip}")

            # Twój pomysł: sprawdzamy zarówno adres źródłowy jak i docelowy!
            ip_match_condition = (self.df['id.orig_h'] == attacker.ip) | (self.df['id.resp_h'] == attacker.ip)

            # Nasz stary pomysł: okno czasowe ataku
            time_condition = (event_timestamps >= attacker.start_ts) & (event_timestamps <= attacker.end_ts)

            # Połączenie obu warunków
            final_condition = ip_match_condition & time_condition

            # Przypisanie etykiety tekstowej (np. 'slowread') i binarnej (1)
            self.df.loc[final_condition, 'label'] = attacker.attack_type
            self.df.loc[final_condition, 'binary_label'] = 1

        print("\n[+] Etykietowanie zakończone!")

        # Wyświetlenie statystyk dokładnie tak jak chciałeś
        print("Rozkład etykiet wieloklasowych (label):")
        print(self.df["label"].value_counts())

        print("\nRozkład etykiet binarnych (binary_label):")
        print(self.df["binary_label"].value_counts())

        self.df.to_csv("dataset_labeled.csv", index=False)
        print("\n[+] Labeled dataset saved!")

    def clean_dataset(self, null_threshold=0.5):
        """
        Usuwa kolumny, w których odsetek brakujących wartości (NaN/Null)
        przekracza podany próg (domyślnie 50%).
        """
        print(f"\n[*] Czyszczenie datasetu: Usuwanie kolumn z brakami danych (próg > {null_threshold * 100}%)...")

        # 1. Obliczanie odsetka braków dla każdej kolumny (wynik od 0.0 do 1.0)
        null_rates = self.df.isnull().mean()

        # 2. Wybieranie nazw kolumn, które przekraczają nasz próg
        columns_to_drop = null_rates[null_rates > null_threshold].index.tolist()

        # 3. Usuwanie i raportowanie
        if columns_to_drop:
            print(f"  -> Usunięto {len(columns_to_drop)} kolumn:")
            for col in columns_to_drop:
                # Wypisuje nazwę kolumny i dokładny procent braków
                print(f"     - {col} ({null_rates[col] * 100:.1f}% nulls)")

            self.df = self.df.drop(columns=columns_to_drop)
        else:
            print("  -> Nie usunięto żadnych kolumn. Wszystkie spełniają kryteria.")

        print(f"  -> Pozostało kolumn w datasecie: {len(self.df.columns)}")

        # Opcjonalnie: usunięcie pojedynczych wierszy, które wciąż mają jakieś nulle
        # print("  -> Usuwanie pojedynczych wierszy z brakującymi danymi...")
        # self.df = self.df.dropna()

    def prepare_data(self):
        print("\n[*] Rygorystyczna selekcja cech (Anti-Freeze Feature Preparation)...")

        # 1. Definiujemy TYLKO te kolumny, które CHCEMY zachować do uczenia modelu.
        # Wszystko inne (w tym ukryte stringi, id, wersje, historia pakietów) zostanie odrzucone.
        keep_numeric = [
            'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'orig_ip_bytes',
            'resp_pkts', 'resp_ip_bytes', 'request_body_len', 'response_body_len',
            'ip_conn_count_10s', 'ip_conn_count_30s', 'ip_avg_duration_30s',
            'ip_sum_orig_bytes_30s', 'ip_sum_resp_bytes_30s', 'ip_avg_orig_pkts_30s',
            'ip_conn_count_60s', 'ip_conn_count_300s', 'ip_avg_duration_60s',
            'ip_avg_duration_300s', 'conn_to_bytes_ratio_30s', 'orig_to_resp_bytes_ratio_30s'
        ]

        keep_categorical = ['conn_state']  # Bardzo stabilna i ważna cecha dla DoS

        # Filtrujemy tylko te kolumny numeryczne, które faktycznie są w tabeli
        existing_numeric = [col for col in keep_numeric if col in self.df.columns]
        existing_categorical = [col for col in keep_categorical if col in self.df.columns]

        # Budujemy czysty DataFrame tylko z bezpiecznych cech
        X = self.df[existing_numeric + existing_categorical].copy()

        # Wybieramy etykietę (Target)
        # Zamiast binarnego 0/1, wybieramy tekstowe etykiety wieloklasowe (normal, slowread, rudy...)
        y = self.df['label'].to_numpy()

        # 2. Uzupełnienie braków w kolumnach numerycznych (zamiana NaN na 0)
        for col in existing_numeric:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)

        # 3. Bezpieczny i szybki One-Hot Encoding tylko dla jednej kolumny
        if existing_categorical:
            print(f"  -> Szybkie kodowanie One-Hot dla: {existing_categorical}")
            # Wymuszamy na pandasie powrót do starego, wydajnego typu int32 bez Nullable extension
            X = pd.get_dummies(X, columns=existing_categorical, drop_first=True, dtype=np.int32)

        print(f"  -> Ostateczna, bezpieczna liczba kolumn cech dla modelu: {X.shape[1]}")
        print("     Kolumny wejściowe:", list(X.columns))

        # 4. Natychmiastowa konwersja na surową macierz NumPy (całkowicie odcina narzut Pandasa)
        self.feature_names = list(X.columns)
        X_clean = X.to_numpy().astype(np.float64)

        # 5. Podział na zbiór treningowy i testowy (teraz wykona się w milisekundę!)
        print("[*] Podział danych na zbiór treningowy i testowy (80/20)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42, stratify=y
        )
        print("[+] Dane zostały podzielone bez zawieszenia systemu!")

    def extract_time_features(self):
        print("\n[*] Budowanie zaawansowanych cech czasowych (Multi-Window Time-Series)...")

        # 1. Konwersja czasu i sortowanie chronologiczne
        self.df['ts_datetime'] = pd.to_datetime(self.df['ts'], format='mixed', errors='coerce')
        self.df = self.df.sort_values(by='ts_datetime').reset_index(drop=True)

        # Wymuszone konwersje typów numerycznych
        print("  -> Standaryzacja typów danych (duration, bytes, pkts)...")
        self.df['duration'] = pd.to_timedelta(self.df['duration'], errors='coerce').dt.total_seconds().fillna(
            0.0).astype(float)
        self.df['orig_bytes'] = pd.to_numeric(self.df['orig_bytes'], errors='coerce').fillna(0.0).astype(float)
        self.df['resp_bytes'] = pd.to_numeric(self.df['resp_bytes'], errors='coerce').fillna(0.0).astype(float)
        self.df['orig_pkts'] = pd.to_numeric(self.df['orig_pkts'], errors='coerce').fillna(0.0).astype(float)

        print("  -> Obliczanie wielookienkowych statystyk kroczących dla adresów IP...")

        # --- OKNA KRÓTKIE (10s i 30s) ---
        self.df['ip_conn_count_10s'] = self.df.groupby('id.orig_h').rolling('10s', on='ts_datetime')[
            'uid'].count().reset_index(level=0, drop=True).values
        self.df['ip_conn_count_30s'] = self.df.groupby('id.orig_h').rolling('30s', on='ts_datetime')[
            'uid'].count().reset_index(level=0, drop=True).values
        self.df['ip_avg_duration_30s'] = self.df.groupby('id.orig_h').rolling('30s', on='ts_datetime')[
            'duration'].mean().reset_index(level=0, drop=True).values
        self.df['ip_sum_orig_bytes_30s'] = self.df.groupby('id.orig_h').rolling('30s', on='ts_datetime')[
            'orig_bytes'].sum().reset_index(level=0, drop=True).values

        # Nowość: Bajty odebrane z serwera w oknie 30s (Locust ma tu wysokie wartości, atak ma zera)
        self.df['ip_sum_resp_bytes_30s'] = self.df.groupby('id.orig_h').rolling('30s', on='ts_datetime')[
            'resp_bytes'].sum().reset_index(level=0, drop=True).values

        # Nowość: Średnia liczba pakietów wysłanych w oknie 30s
        self.df['ip_avg_orig_pkts_30s'] = self.df.groupby('id.orig_h').rolling('30s', on='ts_datetime')[
            'orig_pkts'].mean().reset_index(level=0, drop=True).values

        # --- OKNA DŁUGIE (60s / 1 min oraz 300s / 5 min) ---
        print("  -> Obliczanie kontekstu długoterminowego (1 min i 5 min)...")
        # Liczba połączeń w oknie 1 min i 5 min
        self.df['ip_conn_count_60s'] = self.df.groupby('id.orig_h').rolling('60s', on='ts_datetime')[
            'uid'].count().reset_index(level=0, drop=True).values
        self.df['ip_conn_count_300s'] = self.df.groupby('id.orig_h').rolling('300s', on='ts_datetime')[
            'uid'].count().reset_index(level=0, drop=True).values

        # Średni czas trwania połączeń w oknie 1 min i 5 min
        self.df['ip_avg_duration_60s'] = self.df.groupby('id.orig_h').rolling('60s', on='ts_datetime')[
            'duration'].mean().reset_index(level=0, drop=True).values
        self.df['ip_avg_duration_300s'] = self.df.groupby('id.orig_h').rolling('300s', on='ts_datetime')[
            'duration'].mean().reset_index(level=0, drop=True).values

        # --- WSPÓŁCZYNNIKI I RELACJE (Potężne cechy analityczne) ---
        print("  -> Wyliczanie zaawansowanych współczynników proporcji...")
        # Stosunek liczby połączeń do wysłanych bajtów (Twoja świetna cecha!)
        self.df['conn_to_bytes_ratio_30s'] = self.df['ip_conn_count_30s'] / np.maximum(self.df['ip_sum_orig_bytes_30s'],
                                                                                       1.0)

        # Nowość: Stosunek bajtów wysłanych do odebranych (Atakujący wysyła i nic nie dostaje w zamian)
        self.df['orig_to_resp_bytes_ratio_30s'] = self.df['ip_sum_orig_bytes_30s'] / np.maximum(
            self.df['ip_sum_resp_bytes_30s'], 1.0)

        # 3. Czyszczenie i zabezpieczenie przed NaN
        new_cols = [
            'ip_conn_count_10s', 'ip_conn_count_30s', 'ip_avg_duration_30s', 'ip_sum_orig_bytes_30s',
            'ip_sum_resp_bytes_30s', 'ip_avg_orig_pkts_30s', 'ip_conn_count_60s', 'ip_conn_count_300s',
            'ip_avg_duration_60s', 'ip_avg_duration_300s', 'conn_to_bytes_ratio_30s', 'orig_to_resp_bytes_ratio_30s'
        ]
        self.df[new_cols] = self.df[new_cols].fillna(0.0)

        print(f"[+] Wyekstrahowano zaawansowane cechy. Łączna liczba kolumn w df: {len(self.df.columns)}")

    def train_model(self):
        print("[*] Trenowanie modelu Random Forest...")
        self.model.fit(self.X_train, self.y_train)
        print("[+] Model wytrenowany pomyślnie!")

    def evaluate_model(self):
        print("\n=== EWALUACJA MODELU ===")
        y_pred = self.model.predict(self.X_test)

        print("\nRaport Klasyfikacji:")
        print(classification_report(self.y_test, y_pred))

        class_labels = self.model.classes_

        print("\nGenerowanie wieloklasowej macierzy pomyłek...")
        cm = confusion_matrix(self.y_test, y_pred, labels=class_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels,
                    yticklabels=class_labels)
        plt.ylabel('Rzeczywista klasa')
        plt.xlabel('Przewidziana klasa')
        plt.title('Macierz Błędów - Wieloklasowa Detekcja Ataków')
        plt.tight_layout()
        plt.show()

        # Kod do wklejenia w evaluate_model():
        # Poprawiona sekcja ważności cech
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title("Ważność cech w wykrywaniu Stealthy DoS (Random Forest)")

        # Rysowanie słupków przy użyciu nazw kolumn
        plt.bar(range(len(importances)), importances[indices], align="center", color='skyblue')
        plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig("feature_importance.png")
        print("[+] Wykres ważności cech został zapisany do pliku feature_importance.png")
        plt.show()


def main():
    pipeline = DosDetectionPipeline()
    pipeline.print_data_info()
    pipeline.label_data()
    pipeline.clean_dataset(null_threshold=0.95)

    pipeline.extract_time_features()

    pipeline.prepare_data()
    pipeline.train_model()
    pipeline.evaluate_model()

if __name__ == "__main__":
    main()
