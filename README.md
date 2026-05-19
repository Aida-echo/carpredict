# Classic Car Price Prediction - Streamlit

Aplikasi Streamlit bertema klasik untuk prediksi harga mobil berdasarkan dataset `Car_sales.xlsx`.

## Cara menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File penting

- `app.py` — aplikasi web Streamlit
- `car_price_model.pkl` — model machine learning pipeline
- `requirements.txt` — dependency deploy
- `model_info.json` — metadata fitur dan nilai default

## Catatan

Model memprediksi `Price_in_thousands`, jadi hasil prediksi ditampilkan dalam ribuan USD sesuai format dataset.
