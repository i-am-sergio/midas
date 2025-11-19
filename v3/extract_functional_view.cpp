#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <filesystem>
#include <regex>
#include <sstream>

class FunctionalExtractor {
public:
    FunctionalExtractor(const std::filesystem::path& semanticCsvPath, const std::filesystem::path& projectPath)
        : projectRoot_(projectPath) {
        
        if (!std::filesystem::exists(semanticCsvPath)) {
            throw std::runtime_error("Error: No se pudo encontrar el archivo CSV semántico.");
        }
        if (!std::filesystem::exists(projectRoot_) || !std::filesystem::is_directory(projectRoot_)) {
            throw std::runtime_error("Error: La ruta del proyecto no existe o no es un directorio.");
        }
        
        loadProjectClasses(semanticCsvPath);
    }

    void run(const std::filesystem::path& outputPath) {
        findControllerDependencies(projectRoot_);
        calculateCoOccurrences();
        saveResults(outputPath);
    }

private:
    std::filesystem::path projectRoot_;
    std::set<std::string> allProjectClasses_;
    // Almacena: "NombreControlador" -> {"ClaseA", "ClaseB", "ClaseC"}
    std::map<std::string, std::set<std::string>> controllerDependencies_;
    // Almacena la matriz A_fun: "ClaseA" -> {"ClaseB" -> 5, "ClaseC" -> 2}
    std::map<std::string, std::map<std::string, int>> coOccurrenceMatrix_;

    /**
     * Carga todos los nombres de clases del proyecto desde el CSV semántico
     * para filtrar dependencias (ej. ignorar 'String', 'List', etc.)
     */
    void loadProjectClasses(const std::filesystem::path& csvPath) {
        std::ifstream file(csvPath);
        std::string line;
        std::getline(file, line); // Ignorar cabecera

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string field;
            
            // Parseo simple de CSV: "ClassName","text..."
            std::getline(ss, field, '"'); // Ignora el inicio
            if (std::getline(ss, field, '"')) { // Captura el nombre de la clase
                allProjectClasses_.insert(field);
            }
        }
        std::cout << "Info: Se cargaron " << allProjectClasses_.size() << " nombres de clases del proyecto." << std::endl;
    }

    /**
     * Recorre el directorio del proyecto buscando controladores.
     */
    void findControllerDependencies(const std::filesystem::path& path) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".java") {
                processFile(entry.path());
            }
        }
    }

    /**
     * Procesa un archivo .java. Si es un controlador, extrae sus dependencias.
     */
    void processFile(const std::filesystem::path& filePath) {
        // La "interfaz externa" se redefine como clases en los paquetes web
        if (!isController(filePath.string())) {
            return;
        }
        
        std::ifstream fileStream(filePath);
        if (!fileStream.is_open()) return;

        std::stringstream buffer;
        buffer << fileStream.rdbuf();
        std::string content = buffer.str();
        
        content = removeComments(content);
        std::string controllerName = findEntityName(content);
        if (controllerName.empty()) return;

        std::set<std::string> dependencies = findClassDependencies(content, controllerName);
        controllerDependencies_[controllerName] = dependencies;
    }

    /**
     * Identifica si un archivo es un controlador basado en su ruta.
     */
    bool isController(const std::string& pathString) {
        // Normalizamos los separadores de ruta para la comparación
        std::string normalizedPath = pathString;
        std::replace(normalizedPath.begin(), normalizedPath.end(), '\\', '/');

        return (normalizedPath.find("web/spring") != std::string::npos ||
                normalizedPath.find("web/struts") != std::string::npos);
    }

    /**
     * Extrae todas las clases del proyecto mencionadas en el cuerpo del controlador.
     */
    std::set<std::string> findClassDependencies(const std::string& content, const std::string& selfName) {
        std::set<std::string> dependencies;
        // Regex para encontrar "palabras" que parezcan nombres de clase (inician con mayúscula)
        std::regex typeRegex(R"(\b([A-Z][a-zA-Z0-9_]+)\b)");

        auto words_begin = std::sregex_iterator(content.begin(), content.end(), typeRegex);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::string potentialClass = (*i).str();
            
            // Si es una clase del proyecto Y no es el propio nombre del controlador
            if (potentialClass != selfName && allProjectClasses_.count(potentialClass)) {
                dependencies.insert(potentialClass);
            }
        }
        return dependencies;
    }

    /**
     * Construye la matriz de co-ocurrencia (A_fun) basada en las dependencias
     * de los controladores.
     */
    void calculateCoOccurrences() {
        std::cout << "Info: Calculando co-ocurrencias funcionales..." << std::endl;
        
        // Por cada controlador que encontramos...
        for (const auto& pair : controllerDependencies_) {
            const std::set<std::string>& dependencies = pair.second;
            
            // Iterar sobre todos los pares únicos de dependencias (C_i, C_j)
            for (auto it1 = dependencies.begin(); it1 != dependencies.end(); ++it1) {
                const std::string& classA = *it1;
                for (auto it2 = std::next(it1); it2 != dependencies.end(); ++it2) {
                    const std::string& classB = *it2;
                    
                    // w_ij = w_ij + 1
                    coOccurrenceMatrix_[classA][classB]++;
                    // Aseguramos que la matriz sea simétrica
                    coOccurrenceMatrix_[classB][classA]++;
                }
            }
        }
    }

    // --- Funciones de Utilidad (Reutilizadas de tus scripts) ---

    std::string removeComments(std::string code) {
        code = std::regex_replace(code, std::regex("/\\*[\\s\\S]*?\\*/"), "");
        code = std::regex_replace(code, std::regex("//.*"), "");
        return code;
    }

    std::string findEntityName(const std::string& content) {
        std::smatch match;
        std::regex entityRegex(R"(\b(?:class|interface)\s+([a-zA-Z0-9_]+))");
        if (std::regex_search(content, match, entityRegex)) {
            if (match.size() > 1) {
                return match[1].str();
            }
        }
        return "";
    }
    
    void saveResults(const std::filesystem::path& outputPath) {
        std::ofstream outFile(outputPath);
        if (!outFile.is_open()) {
            throw std::runtime_error("Error: No se pudo crear el archivo CSV de salida.");
        }
        
        outFile << "class,functional_co_occurrences\n";
        
        for (const auto& sourcePair : coOccurrenceMatrix_) {
            outFile << "\"" << sourcePair.first << "\",\"{";
            bool first = true;
            for (const auto& targetPair : sourcePair.second) {
                if (!first) {
                    outFile << ", ";
                }
                // Formato: {'ClaseB': 5, 'ClaseC': 2}
                outFile << "'" << targetPair.first << "': " << targetPair.second;
                first = false;
            }
            outFile << "}\"\n";
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <path_al_semantic_view.csv> <path_del_proyecto_java>" << std::endl;
        return 1;
    }

    std::filesystem::path semanticCsvPath(argv[1]);
    std::filesystem::path projectPath(argv[2]);
    std::filesystem::path canonicalProjectPath;

    try {
        // Validar CSV
        if (!std::filesystem::exists(semanticCsvPath)) {
             std::cerr << "Error: El archivo CSV semántico no existe." << std::endl;
             return 1;
        }

        // Normalizar y validar ruta del proyecto
        canonicalProjectPath = std::filesystem::canonical(projectPath);
        if (!std::filesystem::is_directory(canonicalProjectPath)) {
            std::cerr << "Error: La ruta del proyecto no es un directorio." << std::endl;
            return 1;
        }

        std::string projectName = canonicalProjectPath.filename().string();
        std::string outputFilename = "results/" + projectName + "_fase1_functional_view.csv";
        
        std::cout << "Iniciando extracción de vista funcional para: " << projectName << std::endl;
        FunctionalExtractor extractor(semanticCsvPath, canonicalProjectPath);
        extractor.run(outputFilename);
        
        std::cout << "Proceso completado. Vista funcional guardada en: " << outputFilename << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}