import csv
import requests
import re

# ------------ utilidades de limpieza ------------
def filter_stopwords(words):
    stopwords = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to"}
    return [w for w in words if w.lower() not in stopwords]

def filter_stem_words(words):
    # stemming ultra simple: lowercase + quitar plurales
    return [re.sub(r"s$", "", w.lower()) for w in words]

def filter_params(words):
    # quitar parámetros {xxx} de las URLs
    return [w for w in words if not w.startswith("{")]

def is_http_operation(word):
    return word.lower() in {"get", "post", "put", "delete", "patch", "options", "head"}

# ------------ extracción de APIs desde OpenAPI ------------
def get_apis_from_openapi(json_doc):
    apis = []
    for path, methods in json_doc.get("paths", {}).items():
        for verb, details in methods.items():
            keywords = list(filter(lambda w: w != "", path.split("/")))
            # descripción si existe
            description = details.get("description", "")
            # tokens de url + descripción
            words = " ".join(keywords) + " " + description
            apis.append({
                "url": verb + " " + path,
                "verb": verb,
                "keys": filter_params(keywords),
                "words": words.split(),
                "responses": list(details.get("responses", {}).keys())
            })
    return apis

# ------------ limpieza de datos (sin TFIDF) ------------
def data_clean(apis):
    new_apis = []
    for api in apis:
        words = []
        for word in api["words"]:
            if not is_http_operation(word):
                words.append(word)
        # stopwords y stemming
        no_stop_words = filter_stopwords(words)
        stem_words = filter_stem_words(no_stop_words)
        new_apis.append({
            "url": api["url"],
            "verb": api["verb"],
            "keys": filter_stem_words(api["keys"]),
            "words": stem_words,
            "responses": api["responses"]
        })
    return new_apis

# ------------ guardar en CSV ------------
def save_to_csv(apis, filename="apis.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["url", "verb", "keys", "words", "responses"])
        writer.writeheader()
        for api in apis:
            writer.writerow({
                "url": api["url"],
                "verb": api["verb"],
                "keys": " ".join(api["keys"]),
                "words": " ".join(api["words"]),
                "responses": " ".join(api["responses"])
            })
    print(f"✅ Guardado en {filename}")

# ------------ main ------------
if __name__ == "__main__":
    url = "https://petstore3.swagger.io/api/v3/openapi.json"
    api_doc = requests.get(url).json()

    apis = get_apis_from_openapi(api_doc)
    cleaned_apis = data_clean(apis)
    save_to_csv(cleaned_apis, "jpetstore_apis.csv")
