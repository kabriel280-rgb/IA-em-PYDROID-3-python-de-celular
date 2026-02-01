# IA-em-PYDROID-3-python-de-celular
Chatbot, de python de celular/PYDROID 3/ que possue banco de dados, aprendizado, geração de textos, e usos de IA local


import sqlite3
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
BOT_NAME = "GPT-Lite"
DB_NAME = "chatbot.db"

# ================= BANCO DE DADOS =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk TEXT,
        topic TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_text TEXT,
        bot_text TEXT,
        topic TEXT
    )
    """)

    conn.commit()
    conn.close()

# ================= TREINAMENTO =================
def split_text(text, chunk_size=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks

def train_from_text(text, topic="geral"):
    chunks = split_text(text)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    for chunk in chunks:
        c.execute("INSERT INTO knowledge (chunk, topic) VALUES (?, ?)", (chunk, topic))

    conn.commit()
    conn.close()
    print(f"📚 Treinamento concluído! {len(chunks)} partes salvas no tópico '{topic}'.")

# ================= IA LOCAL =================
def ai_response(user_input, topic):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT chunk FROM knowledge WHERE topic=?", (topic,))
    data = c.fetchall()

    if not data:
        return None

    corpus = [item[0] for item in data] + [user_input]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score > 0.2:
        base_response = corpus[best_match]
        return expand_response(base_response)
    else:
        return None

# ================= EXPANSÃO DE RESPOSTAS =================
def expand_response(base_response):
    openers = [
        "Boa pergunta!",
        "Isso é um assunto interessante.",
        "Vamos explorar isso com mais profundidade.",
        "Ótima escolha de tema.",
        "Essa é uma dúvida muito relevante."
    ]

    closers = [
        "Se quiser, posso detalhar ainda mais esse assunto.",
        "Caso queira exemplos práticos, é só pedir.",
        "Se quiser continuar essa conversa, estou à disposição.",
        "Posso explicar isso passo a passo, se preferir.",
        "Sinta-se à vontade para perguntar mais!"
    ]

    opener = random.choice(openers)
    closer = random.choice(closers)

    expanded = f"{opener}\n\n{base_response}\n\n{closer}"
    return expanded

# ================= COMANDOS =================
def handle_command(command, topic):
    if command == "/ajuda":
        return (
            "📌 Comandos disponíveis:\n"
            "/ajuda - Mostrar ajuda\n"
            "/limpar - Limpar memória\n"
            "/topico <nome> - Definir tópico\n"
            "/topico - Ver tópico atual\n"
            "/treinar - Treinar com texto longo\n"
            "/sair - Encerrar"
        ), topic

    elif command == "/limpar":
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM memory")
        conn.commit()
        conn.close()
        return "🧹 Memória apagada!", topic

    elif command.startswith("/topico"):
        parts = command.split(maxsplit=1)
        if len(parts) == 2:
            topic = parts[1]
            return f"📚 Tópico definido: {topic}", topic
        else:
            return f"📚 Tópico atual: {topic}", topic

    elif command == "/treinar":
        print("📄 Cole o texto longo abaixo. Digite 'FIM' em uma linha separada para terminar:")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "FIM":
                break
            lines.append(line)
        full_text = "\n".join(lines)
        train_from_text(full_text, topic)
        return "✅ Treinamento concluído!", topic

    elif command == "/sair":
        return "👋 Até mais!", topic

    else:
        return "❌ Comando não reconhecido. Digite /ajuda.", topic

# ================= LOOP PRINCIPAL =================
def start_chat():
    print(f"🤖 {BOT_NAME} iniciado!")
    print("Digite /ajuda para ver os comandos.\n")

    topic = "geral"

    while True:
        user_input = input("Você: ")

        if user_input.startswith("/"):
            response, topic = handle_command(user_input, topic)
            print(f"🤖 {response}")
            if user_input == "/sair":
                break
            continue

        response = ai_response(user_input, topic)

        if response:
            print(f"\n🤖 {response}\n")
            save_memory(user_input, response, topic)
        else:
            print("🤖 Ainda não tenho informação suficiente sobre isso nesse tópico.")
            print("💡 Você pode usar /treinar para me ensinar com um texto longo.")

# ================= MEMÓRIA =================
def save_memory(user_text, bot_text, topic):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO memory (user_text, bot_text, topic) VALUES (?, ?, ?)",
              (user_text, bot_text, topic))
    conn.commit()
    conn.close()

# ================= EXECUÇÃO =================
if __name__ == "__main__":
    init_db()
    start_chat()
