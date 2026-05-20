"""
===========================================================================
  Improved Cascade DoS Detector  –  v2 (HistGBM + SMOTE + Rich Features)
===========================================================================
Ulepszenia względem v1 (RandomForest + basic features):

  1. HistGradientBoostingClassifier (sklearn 1.x)
       - Odpowiednik LightGBM: boosting zamiast bagging
       - Natywna obsługa wartości NaN (brak konieczności fillna)
       - Wbudowane early stopping

  2. Bogatsze cechy (packet-level + IP-level + connection history)
       - orig_pkts / resp_pkts, orig_ip_bytes / resp_ip_bytes
       - Bajty na pakiet, ratio odpowiedź/żądanie, przepustowość
       - Flagi historii połączenia: 'W' (okno odczytu – slowread!), 'D', 'R'
       - Okna kroczące 30s i 300s na pakietach i bajtach IP

  3. SMOTE (ręczna implementacja z NearestNeighbors)
       - Stosowany TYLKO do zbioru treningowego modelu forensics
       - Chroni test set przed data leakage
       - Obsługuje NaN przez imputację median przy wyszukiwaniu sąsiadów

  4. Optymalizacja progu Gatekeepera
       - Zamiast stałego 0.35 → szukamy progu max F2-score na zbiorze walidacyjnym
       - F2 = 2*P*R / (4*P + R): faworyzuje recall (krytyczne dla IDS!)

  5. Macierz pomyłek znormalizowana + wykres ważności cech
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


# ────────────────────────────────────────────────────────────
#  SMOTE – ręczna implementacja (brak imbalanced-learn)
# ────────────────────────────────────────────────────────────

def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    target_per_class: int,
    k: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SMOTE generujący próbki syntetyczne dla klas mniejszościowych.

    Parametry
    ---------
    X : tablica treningowa (może zawierać NaN)
    y : etykiety (str lub int)
    target_per_class : docelowa liczba próbek na klasę
    k  : liczba najbliższych sąsiadów dla interpolacji
    random_state : seed losowości

    Zwraca X_resampled, y_resampled (przetasowane).
    """
    rng = np.random.RandomState(random_state)

    # Imputacja median – tylko na potrzeby kNN, oryginalne dane zachowane
    col_medians = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isnan(X), col_medians[np.newaxis, :], X)

    X_parts = [X]
    y_parts = [y]

    for cls in np.unique(y):
        mask = y == cls
        Xc = X[mask]           # oryginalne (z NaN)
        Xc_imp = X_imp[mask]   # imputowane (do kNN)
        current = len(Xc)
        n_gen = max(0, target_per_class - current)

        if n_gen == 0:
            continue

        k_eff = min(k, current - 1)

        if k_eff < 1:
            # Za mało próbek – prosta duplikacja z niewielkim szumem
            synthetic = Xc[rng.randint(0, current, n_gen)]
            noise_std = np.nanstd(Xc, axis=0) * 0.01
            noise = rng.normal(0, np.nan_to_num(noise_std), synthetic.shape)
            X_parts.append(synthetic + noise)
            y_parts.append(np.full(n_gen, cls))
            continue

        nn = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1)
        nn.fit(Xc_imp)
        _, indices = nn.kneighbors(Xc_imp)

        synthetic = np.empty((n_gen, X.shape[1]))
        for i in range(n_gen):
            ref_idx = rng.randint(0, current)
            nn_idx  = rng.choice(indices[ref_idx][1:])   # wyklucz siebie
            alpha   = rng.random()
            synthetic[i] = Xc[ref_idx] + alpha * (Xc[nn_idx] - Xc[ref_idx])

        X_parts.append(synthetic)
        y_parts.append(np.full(n_gen, cls))

    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)

    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


# ────────────────────────────────────────────────────────────
#  Główna klasa detektora
# ────────────────────────────────────────────────────────────

class ImprovedCascadeDetector:
    """
    Dwu-stopniowy kaskadowy detektor ataków DoS klasy slow-rate.

      Stopień 1 – Gatekeeper (binarny):   normal vs atak
      Stopień 2 – Forensics  (wieloklasowy): rudy / slowloris / slowread

    Używa HistGradientBoostingClassifier jako bazowego estymatora
    (natywna obsługa NaN, gradient boosting, szybki).
    """

    LABELS_ORDER = ["normal", "rudy", "slowloris", "slowread"]

    def __init__(self, dataset_path: str = "./dataset_labeled.csv"):
        print("=" * 65)
        print("  Ulepszony System Kaskadowy v2")
        print("  HistGBM + SMOTE + Inżynieria cech pakietowych")
        print("=" * 65)

        self.df: pd.DataFrame = pd.read_csv(dataset_path)
        self.feature_cols: list[str] = []
        self.threshold: float = 0.5  # zostanie zoptymalizowany

        # ── MODEL 1: Gatekeeper ──────────────────────────────
        self.gatekeeper = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            max_leaf_nodes=31,
            class_weight="balanced",
            l2_regularization=0.1,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            verbose=0,
        )

        # ── MODEL 2: Forensics Expert ─────────────────────────
        self.forensics = HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.03,
            max_depth=7,
            min_samples_leaf=5,
            max_leaf_nodes=63,
            class_weight="balanced",
            l2_regularization=0.05,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            verbose=0,
        )

    # ────────────────────────────────────────────────────────
    #  Inżynieria cech
    # ────────────────────────────────────────────────────────

    def _cast_numeric(self, cols: list[str]) -> None:
        for c in cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

    def extract_features(self) -> None:
        print("\n[*] Inżynieria cech…")
        df = self.df

        # ── Czas ─────────────────────────────────────────────
        df["ts_dt"]     = pd.to_datetime(df["ts"], format="mixed", errors="coerce")
        df               = df.sort_values("ts_dt").reset_index(drop=True)
        df["duration_s"] = (
            pd.to_timedelta(df["duration"], errors="coerce")
            .dt.total_seconds()
        )  # NaN zachowane – HistGBM sobie z nimi poradzi

        # ── Rzutowanie numeryczne ─────────────────────────────
        self._cast_numeric([
            "orig_bytes", "resp_bytes",
            "orig_pkts",  "resp_pkts",
            "orig_ip_bytes", "resp_ip_bytes",
            "missed_bytes",
            "request_body_len", "response_body_len",
            "id.orig_p", "id.resp_p",
        ])

        # ── Cechy pakietowe (KLUCZOWE dla slow-rate) ──────────
        eps = 1e-9  # unikamy dzielenia przez zero / NaN

        # Bajty na pakiet → małe wartości = powolny transfer
        df["orig_bytes_per_pkt"] = df["orig_ip_bytes"] / (df["orig_pkts"] + eps)
        df["resp_bytes_per_pkt"] = df["resp_ip_bytes"] / (df["resp_pkts"] + eps)

        # Stosunek ruchu odpowiedzi do żądania
        # Normal: serwer odsyła dużo (>1), ataki: ~1 lub <1
        df["resp_to_orig_bytes_ratio"] = df["resp_ip_bytes"] / (df["orig_ip_bytes"] + eps)
        df["resp_to_orig_pkts_ratio"]  = df["resp_pkts"]    / (df["orig_pkts"]    + eps)

        # Przepustowość (B/s)
        df["orig_bytes_per_sec"] = df["orig_ip_bytes"] / (df["duration_s"] + eps)
        df["resp_bytes_per_sec"] = df["resp_ip_bytes"] / (df["duration_s"] + eps)

        # Pakiety na sekundę
        df["orig_pkts_per_sec"] = df["orig_pkts"] / (df["duration_s"] + eps)
        df["resp_pkts_per_sec"] = df["resp_pkts"] / (df["duration_s"] + eps)

        # Czas trwania × pakiety (slow: długa sesja, mało pakietów)
        df["duration_x_pkts"]   = df["duration_s"] * df["orig_pkts"]
        df["duration_x_bytes"]  = df["duration_s"] * df["orig_ip_bytes"]

        # Proporcja missed_bytes (utracone dane → slow read!)
        df["missed_bytes_ratio"] = df["missed_bytes"] / (
            df["orig_ip_bytes"] + df["resp_ip_bytes"] + eps
        )

        # ── Flagi historii połączenia ─────────────────────────
        hist = df["history"].fillna("")
        df["hist_len"]     = hist.str.len().astype(float)
        df["hist_has_W"]   = hist.str.contains("W",  regex=False).astype(float)  # slow read!
        df["hist_has_D"]   = hist.str.contains("D",  regex=False).astype(float)
        df["hist_has_R"]   = hist.str.contains("R",  regex=False).astype(float)
        df["hist_has_S"]   = hist.str.contains("S",  regex=False).astype(float)
        df["hist_has_A"]   = hist.str.contains("A",  regex=False).astype(float)
        df["hist_has_F"]   = hist.str.contains("F",  regex=False).astype(float)
        df["hist_is_one_sided"] = hist.str.startswith("^").astype(float)  # ^a = brak SYN od serwera

        # ── Cechy metody HTTP ─────────────────────────────────
        # Rudy = powolny POST → kluczowe!
        method = df["method"].astype(str).str.upper()
        df["is_post"] = (method == "POST").astype(float)
        df["is_get"]  = (method == "GET").astype(float)

        # ── Okna kroczące 30s i 300s ─────────────────────────
        print("    [*] Obliczanie okien kroczących (może chwilę potrwać)…")
        for win, suf in [("30s", "30s"), ("300s", "300s")]:
            g = df.groupby("id.orig_h", group_keys=False)

            df[f"ip_conn_count_{suf}"] = (
                g.rolling(win, on="ts_dt")["uid"]
                .count()
                .reset_index(level=0, drop=True)
                .values
            )
            df[f"ip_sum_orig_bytes_{suf}"] = (
                g.rolling(win, on="ts_dt")["orig_ip_bytes"]
                .sum()
                .reset_index(level=0, drop=True)
                .values
            )
            df[f"ip_sum_resp_bytes_{suf}"] = (
                g.rolling(win, on="ts_dt")["resp_ip_bytes"]
                .sum()
                .reset_index(level=0, drop=True)
                .values
            )
            df[f"ip_sum_pkts_{suf}"] = (
                g.rolling(win, on="ts_dt")["orig_pkts"]
                .sum()
                .reset_index(level=0, drop=True)
                .values
            )
            df[f"ip_avg_duration_{suf}"] = (
                g.rolling(win, on="ts_dt")["duration_s"]
                .mean()
                .reset_index(level=0, drop=True)
                .values
            )

        # Pochodne okien
        df["ip_avg_bytes_per_conn_30s"] = (
            df["ip_sum_orig_bytes_30s"] / (df["ip_conn_count_30s"] + eps)
        )
        df["ip_avg_pkts_per_conn_30s"] = (
            df["ip_sum_pkts_30s"] / (df["ip_conn_count_30s"] + eps)
        )
        df["ip_bytes_ratio_30s_300s"] = (
            df["ip_sum_orig_bytes_30s"] / (df["ip_sum_orig_bytes_300s"] + eps)
        )

        self.df = df
        print(f"    [+] Gotowe. Łącznie kolumn w DataFrame: {len(df.columns)}")

    # ────────────────────────────────────────────────────────
    #  Budowa macierzy cech
    # ────────────────────────────────────────────────────────

    def _build_feature_matrix(self) -> np.ndarray:
        df = self.df

        numeric_cols = [
            # Podstawowe
            "duration_s",
            "orig_bytes", "resp_bytes",
            "orig_pkts",  "resp_pkts",
            "orig_ip_bytes", "resp_ip_bytes",
            "missed_bytes",
            "request_body_len", "response_body_len",
            # Pakietowe
            "orig_bytes_per_pkt", "resp_bytes_per_pkt",
            "resp_to_orig_bytes_ratio", "resp_to_orig_pkts_ratio",
            "orig_bytes_per_sec", "resp_bytes_per_sec",
            "orig_pkts_per_sec", "resp_pkts_per_sec",
            "duration_x_pkts", "duration_x_bytes",
            "missed_bytes_ratio",
            # Historia
            "hist_len",
            "hist_has_W", "hist_has_D", "hist_has_R",
            "hist_has_S", "hist_has_A", "hist_has_F",
            "hist_is_one_sided",
            # HTTP
            "is_post", "is_get",
            # Porty
            "id.orig_p", "id.resp_p",
            # Okna
            "ip_conn_count_30s",  "ip_sum_orig_bytes_30s", "ip_sum_resp_bytes_30s",
            "ip_sum_pkts_30s",    "ip_avg_duration_30s",
            "ip_conn_count_300s", "ip_avg_duration_300s",
            "ip_avg_bytes_per_conn_30s", "ip_avg_pkts_per_conn_30s",
            "ip_bytes_ratio_30s_300s",
        ]

        X = df[numeric_cols].copy()

        # One-hot: conn_state (OTH jest silnie skorelowany z atakami!)
        if "conn_state" in df.columns:
            cs_dummies = pd.get_dummies(df["conn_state"], prefix="cs", dtype=float)
            X = pd.concat([X, cs_dummies], axis=1)

        # One-hot: protokół
        if "proto" in df.columns:
            proto_dummies = pd.get_dummies(df["proto"], prefix="proto", dtype=float)
            X = pd.concat([X, proto_dummies], axis=1)

        self.feature_cols = X.columns.tolist()
        return X.to_numpy(dtype=np.float64)

    # ────────────────────────────────────────────────────────
    #  Optymalizacja progu (Gatekeeper)
    # ────────────────────────────────────────────────────────

    def _optimize_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        """
        Szuka progu maksymalizującego F_beta (beta=2) na zbiorze walidacyjnym.
        F2 faworyzuje recall – minimalizuje przeoczone ataki (false negatives).
        """
        proba = self.gatekeeper.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, proba)

        beta = 2.0
        denom = beta**2 * precision + recall
        f_beta = np.where(
            denom > 0,
            (1 + beta**2) * precision * recall / denom,
            0.0,
        )

        best_idx = int(np.argmax(f_beta[:-1]))
        best_thr = float(thresholds[best_idx])
        best_f2  = float(f_beta[best_idx])

        print(f"    [+] Optymalny próg: {best_thr:.4f}  (F2 = {best_f2:.4f})")
        return best_thr

    # ────────────────────────────────────────────────────────
    #  Trening
    # ────────────────────────────────────────────────────────

    def prepare_and_train(self) -> None:
        print("\n[*] Przygotowanie danych i trening modeli…")

        X     = self._build_feature_matrix()
        y_bin = self.df["binary_label"].to_numpy()
        y_mul = self.df["label"].to_numpy()

        print(f"    Macierz cech: {X.shape[0]} próbek × {X.shape[1]} cech")

        # ── Podział: train / test ─────────────────────────────
        (X_train, X_test,
         y_tr_b,  y_te_b,
         y_tr_m,  y_te_m) = train_test_split(
            X, y_bin, y_mul,
            test_size=0.20,
            random_state=42,
            stratify=y_mul,
        )

        # ── Podział: train / validation (do opt. progu) ───────
        (X_tr, X_val,
         y_tr_b2, y_val_b,
         y_tr_m2, _) = train_test_split(
            X_train, y_tr_b, y_tr_m,
            test_size=0.15,
            random_state=42,
            stratify=y_tr_b,
        )

        self.X_test  = X_test
        self.y_te_m  = y_te_m

        # ── Model 1: Gatekeeper ───────────────────────────────
        print("\n  [ETAP 1] Gatekeeper (binarny, HistGBM)…")
        self.gatekeeper.fit(X_tr, y_tr_b2)
        print(f"    Liczba iteracji boosting: {self.gatekeeper.n_iter_}")

        # AUC na walidacji
        gk_proba_val = self.gatekeeper.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val_b, gk_proba_val)
        print(f"    ROC-AUC (walidacja): {auc_val:.4f}")

        # Optymalizacja progu
        self.threshold = self._optimize_threshold(X_val, y_val_b)

        # ── Model 2: Forensics + SMOTE ────────────────────────
        print("\n  [ETAP 2] Forensics Expert (wieloklasowy, HistGBM + SMOTE)…")

        atk_mask = y_tr_b2 == 1
        X_atk    = X_tr[atk_mask]
        y_atk    = y_tr_m2[atk_mask]

        print("    Dystrybucja PRZED SMOTE:")
        for cls, cnt in zip(*np.unique(y_atk, return_counts=True)):
            print(f"      {cls:<12}: {cnt:>5}")

        # Cel: max(istniejąca_klasa, 600) dla każdej klasy
        max_existing = max(
            cnt for cnt in [np.sum(y_atk == c) for c in np.unique(y_atk)]
        )
        smote_target = max(max_existing, 600)

        X_atk_sm, y_atk_sm = smote_oversample(
            X_atk, y_atk,
            target_per_class=smote_target,
            k=5,
            random_state=42,
        )

        print("    Dystrybucja PO SMOTE:")
        for cls, cnt in zip(*np.unique(y_atk_sm, return_counts=True)):
            print(f"      {cls:<12}: {cnt:>5}")

        self.forensics.fit(X_atk_sm, y_atk_sm)
        print(f"    Liczba iteracji boosting: {self.forensics.n_iter_}")
        print("\n  [+] Oba modele wytrenowane pomyślnie!")

    # ────────────────────────────────────────────────────────
    #  Ewaluacja
    # ────────────────────────────────────────────────────────

    def evaluate(self) -> None:
        print("\n" + "=" * 65)
        print("  EWALUACJA KASKADY v2")
        print("=" * 65)

        # Krok 1: Gatekeeper
        proba_bin = self.gatekeeper.predict_proba(self.X_test)[:, 1]
        pred_bin  = (proba_bin >= self.threshold).astype(int)

        final_pred = np.array(["normal"] * len(self.X_test), dtype=object)
        suspicious = np.where(pred_bin == 1)[0]

        n_susp = len(suspicious)
        pct    = 100.0 * n_susp / len(self.X_test)
        print(f"\n  Gatekeeper wyłapał {n_susp}/{len(self.X_test)} ({pct:.1f}%) połączeń jako podejrzane")

        # Krok 2: Forensics
        if n_susp > 0:
            final_pred[suspicious] = self.forensics.predict(self.X_test[suspicious])

        # ── Raport ───────────────────────────────────────────
        print("\n  Raport klasyfikacji (Kaskada v2 – HistGBM + SMOTE):")
        print(classification_report(
            self.y_te_m, final_pred,
            labels=self.LABELS_ORDER,
            digits=4,
        ))

        # ── Macierz pomyłek ───────────────────────────────────
        cm      = confusion_matrix(self.y_te_m, final_pred, labels=self.LABELS_ORDER)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig = plt.figure(figsize=(18, 7))
        fig.suptitle(
            "Kaskadowy Detektor DoS v2  –  HistGradientBoosting + SMOTE",
            fontsize=14, fontweight="bold", y=1.01,
        )
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.45)

        # Panel 1: wartości bezwzględne
        ax1 = fig.add_subplot(gs[0])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=self.LABELS_ORDER,
            yticklabels=self.LABELS_ORDER,
            ax=ax1,
        )
        ax1.set_title("Liczby bezwzględne", fontsize=11)
        ax1.set_ylabel("Rzeczywista klasa")
        ax1.set_xlabel("Przewidziana klasa")

        # Panel 2: znormalizowana (per-row recall)
        ax2 = fig.add_subplot(gs[1])
        sns.heatmap(
            cm_norm, annot=True, fmt=".1%", cmap="Purples",
            xticklabels=self.LABELS_ORDER,
            yticklabels=self.LABELS_ORDER,
            ax=ax2,
            vmin=0, vmax=1,
        )
        ax2.set_title("Znormalizowana (recall per class)", fontsize=11)
        ax2.set_ylabel("")
        ax2.set_xlabel("Przewidziana klasa")

        # Panel 3: ważność cech (Gatekeeper)
        ax3 = fig.add_subplot(gs[2])
        if hasattr(self.gatekeeper, "feature_importances_"):
            importances = self.gatekeeper.feature_importances_
            top_n = 20
            idx   = np.argsort(importances)[-top_n:]
            names = [self.feature_cols[i] for i in idx]
            vals  = importances[idx]

            colors = ["#6a0dad" if v >= vals[-5] else "#c299e0" for v in vals]
            ax3.barh(names, vals, color=colors)
            ax3.set_title(f"Top {top_n} cech – Gatekeeper", fontsize=11)
            ax3.set_xlabel("Ważność (gain)")
            ax3.tick_params(axis="y", labelsize=8)

        plt.tight_layout()
        out_path = "cascade_confusion_matrix_v2.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n  [+] Wykres zapisany: {out_path}")
        plt.show()

        # ── Podsumowanie poprawy vs v1 ────────────────────────
        self._print_improvement_summary(cm)

    def _print_improvement_summary(self, cm: np.ndarray) -> None:
        """Porównuje recall v2 z wartościami z v1 (odczytanymi z macierzy)."""
        v1_cm = np.array([
            [13474,    12,    11,   42],
            [   25,    69,     3,    2],
            [   20,     6,   188,    1],
            [  141,     0,     0,  304],
        ])

        print("\n  ┌─────────────────────────────────────────────┐")
        print("  │  Porównanie Recall: v1 (RF) vs v2 (HistGBM) │")
        print("  ├────────────┬──────────────┬─────────────────┤")
        print("  │  Klasa     │  v1 Recall   │  v2 Recall      │")
        print("  ├────────────┼──────────────┼─────────────────┤")

        for i, lbl in enumerate(self.LABELS_ORDER):
            v1_rec = v1_cm[i, i] / max(v1_cm[i].sum(), 1)
            v2_rec = cm[i, i]    / max(cm[i].sum(),    1)
            delta  = v2_rec - v1_rec
            arrow  = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            print(
                f"  │ {lbl:<10} │  {v1_rec*100:>6.1f}%     │"
                f"  {v2_rec*100:>6.1f}%  {arrow}{abs(delta)*100:+.1f}pp  │"
            )

        print("  └────────────┴──────────────┴─────────────────┘")


# ────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────

def main() -> None:
    detector = ImprovedCascadeDetector(dataset_path="./dataset_labeled.csv")
    detector.extract_features()
    detector.prepare_and_train()
    detector.evaluate()


if __name__ == "__main__":
    main()