import javalang
import os
import csv

def extract_classes(java_code):
    """
    Extrae los nombres de clases y el paquete donde están.
    """
    tree = javalang.parse.parse(java_code)
    package_name = None
    classes = []

    # Obtener el nombre del paquete si existe
    if tree.package:
        package_name = tree.package.name

    # Obtener las clases declaradas
    for path, cls in tree.filter(javalang.tree.ClassDeclaration):
        classes.append((package_name, cls.name))

    return classes


def parse_project(src_folder, output_csv="classes.csv"):
    """
    Recorre un proyecto Java y guarda en un CSV los paquetes y clases encontradas.
    """
    all_classes = []

    for root, _, files in os.walk(src_folder):
        for f in files:
            if f.endswith(".java"):
                with open(os.path.join(root, f), encoding="utf-8") as fp:
                    code = fp.read()
                try:
                    classes = extract_classes(code)
                    all_classes.extend(classes)
                except Exception as e:
                    print(f"Error en {f}: {e}")

    # Guardar en CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["package", "class_name"])
        writer.writerows(all_classes)

    print(f"CSV generado: {output_csv} con {len(all_classes)} clases")


if __name__ == "__main__":
    project_path = "monoliths/jPetStore/src"  # cambia la ruta según tu proyecto
    parse_project(project_path, "dataset_jpetstore_classes.csv")
