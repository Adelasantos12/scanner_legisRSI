import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

######################################################################
# CONFIGURACIÓN GENERAL
######################################################################

# Ruta base del repo
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Archivos de entrenamiento por país
# 👉 Aquí agregas más países en el futuro. Basta con poner más CSVs.
country_datasets = [
    os.path.join(BASE_DIR, "data", "training_data_mexico_legal_extended.csv"),
    # os.path.join(BASE_DIR, "data", "training_data_colombia_legal_extended.csv"),
    # os.path.join(BASE_DIR, "data", "training_data_peru_legal_extended.csv"),
]

# Columna de texto y etiqueta
TEXT_COL = "texto_normalizado"
LABEL_COL = "funcion_RSI"

# Carpeta de salida del modelo
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "bert_rsi_latam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modelo base multilingüe
MODEL_NAME = "bert-base-multilingual-cased"

# Hiperparámetros básicos
EPOCHS = 4
BATCH_SIZE = 8
LR = 2e-5
MAX_LEN = 256
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


######################################################################
# UTILIDADES: CARGA Y PREPARACIÓN DEL DATASET
######################################################################

def load_all_countries(datasets_paths):
    """
    Carga todos los CSV declarados en country_datasets,
    concatena y devuelve un único DataFrame.

    Importante:
    - Sólo usamos 'texto_normalizado' como feature de entrada.
    - No usamos atributos mexicanos específicos como 'dof_ref' o 'actor_principal'
      para evitar sobreajuste a México. Eso hace el modelo portable.
    """
    frames = []
    for path in datasets_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["__source_file__"] = os.path.basename(path)
            frames.append(df)
        else:
            print(f"⚠️ Advertencia: no se encontró {path}, se omite.")
    if not frames:
        raise RuntimeError("No se cargó ningún dataset. Revisa las rutas en country_datasets.")
    all_df = pd.concat(frames, ignore_index=True)

    # Drop filas que no tengan texto o etiqueta
    all_df = all_df.dropna(subset=[TEXT_COL, LABEL_COL])
    all_df = all_df[all_df[TEXT_COL].astype(str).str.strip() != ""]
    all_df = all_df[all_df[LABEL_COL].astype(str).str.strip() != ""]
    all_df = all_df.reset_index(drop=True)

    return all_df


def stratified_split(df, test_size=0.2):
    """
    Partición estratificada para que todas las funciones RSI
    estén presentes en train y valid.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[LABEL_COL],
        random_state=RANDOM_SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


######################################################################
# DATASET Y COLLATE
######################################################################

class LegalRSIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = str(self.texts[idx])

        enc = self.tokenizer(
            t,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item


######################################################################
# ENTRENAMIENTO
######################################################################

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Entrenando", leave=False):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        refs = labels.detach().cpu().numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(refs))

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1


@torch.no_grad()
def eval_epoch(model, dataloader, device, id2label):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Validando", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        refs = labels.detach().cpu().numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(refs))

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print("\n===== Reporte clasificación (validación) =====")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=3
    ))

    return avg_loss, acc, f1


######################################################################
# MAIN
######################################################################

def main():
    print("📥 Cargando datasets declarados...")
    df = load_all_countries(country_datasets)

    # -----------------------------------------------------------------
    # Etiquetador (funcion_RSI): creamos encoding estable
    # Ejemplo de clases esperadas:
    # vigilancia, deteccion, notificacion, respuesta,
    # coordinacion, preparacion, poe, recursos_humanos,
    # legislacion_financiamiento, amenazas_especificas,
    # comunicacion_riesgo
    # -----------------------------------------------------------------
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df[LABEL_COL])

    id2label = {i: lab for i, lab in enumerate(label_encoder.classes_)}
    label2id = {lab: i for i, lab in id2label.items()}

    print("Clases RSI:", id2label)

    # -----------------------------------------------------------------
    # Partición train / valid estratificada
    # -----------------------------------------------------------------
    train_df, val_df = stratified_split(df, test_size=0.2)

    # -----------------------------------------------------------------
    # Tokenizador y datasets
    # -----------------------------------------------------------------
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = LegalRSIDataset(
        texts=train_df[TEXT_COL],
        labels=train_df["label_id"],
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = LegalRSIDataset(
        texts=val_df[TEXT_COL],
        labels=val_df["label_id"],
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------------------
    # Modelo
    # -----------------------------------------------------------------
    num_labels = len(label_encoder.classes_)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("💻 Usando dispositivo:", device)
    model.to(device)

    # -----------------------------------------------------------------
    # Optimizador y scheduler
    # -----------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # -----------------------------------------------------------------
    # Loop de entrenamiento
    # -----------------------------------------------------------------
    best_val_f1 = -1.0
    history = []

    for epoch in range(EPOCHS):
        print(f"\n🌎 Época {epoch+1}/{EPOCHS}")

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )

        val_loss, val_acc, val_f1 = eval_epoch(
            model, val_loader, device, id2label
        )

        print(f"[Época {epoch+1}] train_loss={train_loss:.4f} train_f1={train_f1:.3f} val_loss={val_loss:.4f} val_f1={val_f1:.3f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1_macro": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1
        })

        # Guardar el mejor modelo según F1 macro en validación
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(model, tokenizer, label_encoder, OUTPUT_DIR)
            print(f"💾 Modelo actualizado (mejor F1 valid = {best_val_f1:.3f})")

    # Guardar historial de entrenamiento
    hist_path = os.path.join(OUTPUT_DIR, "training_history.json")
    pd.DataFrame(history).to_json(hist_path, orient="records", force_ascii=False, indent=2)
    print(f"📊 Historial guardado en {hist_path}")

    print("\n✅ Entrenamiento terminado.")


######################################################################
# GUARDADO DEL MODELO
######################################################################

def save_model(model, tokenizer, label_encoder, out_dir):
    """
    Guarda:
    - pesos del modelo
    - tokenizer
    - mapping id<->label (para inferencia)
    """
    os.makedirs(out_dir, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    labels_path = os.path.join(out_dir, "label_mapping.json")
    mapping = {
        "id2label": {str(i): lab for i, lab in enumerate(label_encoder.classes_)},
        "label2id": {lab: int(i) for i, lab in enumerate(label_encoder.classes_)}
    }

    import json
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"💾 Modelo y etiquetas guardadas en {out_dir}")


######################################################################
# ENTRYPOINT
######################################################################

if __name__ == "__main__":
    main()
