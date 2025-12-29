## Pruebas de Funcionamiento de Modelos de Tópicos

### Resumen general
Este repositorio contiene dos scripts independientes que sirven como **pruebas mínimas de funcionamiento** para cuatro modelos de tópicos aplicados al dataset **20Newsgroups**:

1. **`LDA_FASTopic_ECRTM.ipynb`**
   Ejecuta tres modelos clásicos y neurales rápidos:
   - LDA (Gensim)
   - FASTopic (TopMost)
   - ECRTM (TopMost)

2. **`TNTM.ipynb`**
   Entrena el modelo avanzado **TNTM** utilizando embeddings precalculados de BERT.

Ambos scripts están diseñados para ejecutarse de forma aislada y verificar que los modelos se inicializan, entrenan y producen resultados mínimos sin errores.

### Requisitos comunes
- Python 3.9 o superior.

### Scripts individuales

#### 1. LDA_FASTopic_ECRTM.ipynb – Prueba de LDA, FASTopic y ECRTM
**Objetivo**: Verificar el funcionamiento básico de tres modelos sobre 1000 documentos.

**Requisitos específicos**:
```bash
pip install torch scikit-learn gensim topmost
```

**Configuración**:
- Documentos: 1000
- Tópicos: 10
- Vocabulario (TopMost): 2000 palabras
- Dispositivo: CPU (cambiar a `"cuda"` si hay GPU disponible)

**Salida**: Impresión en consola de las 10 palabras más representativas por tópico para cada modelo.

#### 2. TNTM.ipynb – Prueba de TNTM
**Objetivo**: Entrenar correctamente TNTM con embeddings de BERT sobre 600 documentos, incluyendo splits de train/val/test.

**Requisitos específicos**:
```bash
pip install pickle torch pandas tqdm numpy octis TNTM_SentenceTransformer
```
- Archivo requerido: `TNTM/Data/DataResults_BERT/cleaned_embedding_df_20ng_BERT.pickle` (embeddings precalculados de BERT).

**Configuración**:
- Documentos: 600 (configurable)
- Tópicos: 20
- Tasas de aprendizaje: Encoder 1e-3, Decoder 1e-3
- Semilla: 42

**Salidas**:
- Modelo entrenado: `example/20_topics_600docs/model.pt`
- Conjunto de prueba: `example/20_topics_600docs/test_set_60docs.pickle`

**Notas técnicas**:
- Incluye parches para compatibilidad con tensores dispersos y evitar conjuntos internos vacíos.
- Conversión automática de sparse a dense cuando es necesario.

### Notas generales
- Ambos scripts son independientes y están pensados como pruebas rápidas de funcionamiento.
- Los resultados pueden variar ligeramente entre ejecuciones debido a inicializaciones aleatorias (excepto donde se fija la semilla).
- Ideales para verificar la instalación correcta de las librerías y el entorno antes de experimentos más grandes.

