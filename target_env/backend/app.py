from flask import Flask, request, Response

app = Flask(__name__)

# --- WYGLĄD STRONY GŁÓWNEJ (HTML + CSS) ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Serwer Celu - Testbed DoS</title>
    <style>
        body {
            font-family: 'Courier New', Courier, monospace;
            background-color: #121212;
            color: #00ff00;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #1e1e1e;
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 0 25px rgba(0, 255, 0, 0.15);
            border: 1px solid #333;
            text-align: center;
            max-width: 600px;
            width: 90%;
        }
        h1 {
            color: #ffffff;
            border-bottom: 2px solid #00ff00;
            padding-bottom: 15px;
            margin-top: 0;
        }
        p {
            font-size: 1.1rem;
            color: #cccccc;
        }
        .status {
            display: inline-block;
            padding: 8px 20px;
            background-color: #28a745;
            color: white;
            border-radius: 25px;
            font-weight: bold;
            margin: 20px 0;
            letter-spacing: 1px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
            70% { box-shadow: 0 0 0 15px rgba(40, 167, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
        }
        .form-box {
            margin-top: 30px;
            padding: 20px;
            background-color: #252525;
            border-radius: 8px;
            border: 1px dashed #555;
        }
        input[type="text"] {
            padding: 12px;
            border: 1px solid #444;
            background-color: #111;
            color: #00ff00;
            border-radius: 4px;
            width: calc(100% - 26px);
            margin-bottom: 15px;
            font-family: inherit;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #00ff00;
        }
        button {
            padding: 12px 25px;
            background-color: #00ff00;
            color: #000;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            font-size: 1rem;
            font-family: inherit;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #00cc00;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Serwer Docelowy</h1>
        <p>Środowisko testowe dla ataków typu Low-Rate Stealthy DoS.</p>
        <div class="status">STATUS: ONLINE</div>
        
        <div class="form-box">
            <h3 style="color: #fff; margin-top: 0;">Testowy formularz POST</h3>
            <p style="font-size: 0.9rem;">(Cel dla ataku R.U.D.Y.)</p>
            <form action="/rudy" method="POST">
                <input type="text" name="data" placeholder="Wprowadź przykładowe dane...">
                <button type="submit">Wyślij POST</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

# Endpoint dla Slowloris i ogólnego ruchu
@app.route('/', methods=['GET'])
def index():
    return HTML_PAGE

# Endpoint dla ataku Shrew (wymusza długi transfer TCP - bez UI, zwraca gołe dane)
@app.route('/download', methods=['GET'])
def download():
    def generate():
        for _ in range(5000):
            yield b"A" * 10240
    return Response(generate(), mimetype='application/octet-stream')

# Endpoint dla ataku RUDY (oczekuje na dane POST)
@app.route('/rudy', methods=['POST'])
def rudy_target():
    data = request.get_data()
    # Ładny widok również po wysłaniu formularza
    return f"""
    <html>
    <body style="background-color: #121212; color: #00ff00; font-family: 'Courier New', Courier, monospace; text-align: center; padding-top: 100px;">
        <h2 style="color: #fff;">Zakończono przetwarzanie</h2>
        <p style="font-size: 1.2rem;">Odebrano dokładnie <b style="color: #fff;">{len(data)}</b> bajtów danych z żądania POST.</p>
        <br><br>
        <a href="/" style="color: #00ff00; text-decoration: none; border: 1px solid #00ff00; padding: 10px 20px; border-radius: 5px;">&laquo; Powrót do panelu</a>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)