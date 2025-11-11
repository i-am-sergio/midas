#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <regex>
#include <sstream>

// Estructura para almacenar la información de cada clase leída del CSV
struct ClassInfo {
    std::string filename;
    std::string simpleName;
    std::string attributesStr;
    std::string methodsStr;
    std::string fullContent; // Contenido completo del archivo .java
};

class RelationAnalyzer {
public:
    RelationAnalyzer(const std::filesystem::path& csvPath, const std::filesystem::path& projectPath)
        : projectRoot_(projectPath) {
        
        loadDataFromCsv(csvPath);
        loadJavaFileContents();
    }

    void analyzeAndSave(const std::filesystem::path& outputPath) {
        std::map<std::string, std::map<std::string, int>> relationScores;

        for (const auto& pair : classes_) {
            const std::string& sourceClassName = pair.first;
            const ClassInfo& sourceClassInfo = pair.second;
            
            analyzeInheritance(sourceClassName, sourceClassInfo, relationScores);
            analyzeSignatures(sourceClassName, sourceClassInfo, relationScores);
            analyzeBodyInstantiations(sourceClassName, sourceClassInfo, relationScores);
        }
        
        saveResults(outputPath, relationScores);
    }

private:
    std::filesystem::path projectRoot_;
    std::map<std::string, ClassInfo> classes_;

    // Puntuaciones
    const int INHERITANCE_SCORE = 15;
    const int INSTANTIATION_SCORE = 8;
    const int ATTRIBUTE_SCORE = 5;
    const int METHOD_SIG_SCORE = 3;

    void loadDataFromCsv(const std::filesystem::path& csvPath) {
        std::ifstream file(csvPath);
        if (!file.is_open()) {
            throw std::runtime_error("Error: No se pudo abrir el archivo CSV de entrada.");
        }

        std::string line;
        std::getline(file, line); // Ignorar cabecera

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string field;
            std::vector<std::string> fields;
            
            // Simple CSV parsing for quoted fields
            while (std::getline(ss, field, '"')) {
                if (!field.empty() && field != ",") {
                    fields.push_back(field);
                }
            }

            if (fields.size() >= 4) {
                ClassInfo info;
                info.filename = fields[0];
                info.simpleName = fields[1];
                info.attributesStr = fields[2];
                info.methodsStr = fields[3];
                classes_[info.simpleName] = info;
            }
        }
    }
    
    void loadJavaFileContents() {
        for (auto& pair : classes_) {
            std::ifstream javaFile(pair.second.filename); // Abrir el archivo .java
            if(javaFile.is_open()){ // Verificar que se abrió correctamente
                std::stringstream buffer; // Usar stringstream para leer todo el contenido
                buffer << javaFile.rdbuf(); // Leer el contenido completo
                pair.second.fullContent = buffer.str(); // Almacenar el contenido en ClassInfo
            }
        }
    }

    void analyzeInheritance(const std::string& sourceClass, const ClassInfo& info, std::map<std::string, std::map<std::string, int>>& scores) {
        std::regex inheritanceRegex(R"(\b(?:extends|implements)\s+((?:\w+(?:,\s*\w+)*)))");
        std::smatch match;

        if (std::regex_search(info.fullContent, match, inheritanceRegex) && match.size() > 1) {
            std::string relatedClassesStr = match[1].str();
            std::cout << "Inheritance found in " << sourceClass << ": " << relatedClassesStr << std::endl;
            std::stringstream ss(relatedClassesStr);
            std::string className;
            while(ss >> className){
                if(className.back() == ',') className.pop_back();
                if(classes_.count(className)){
                     scores[sourceClass][className] += INHERITANCE_SCORE;
                }
            }
        }
    }

    void analyzeSignatures(const std::string& sourceClass, const ClassInfo& info, std::map<std::string, std::map<std::string, int>>& scores) {
        std::string combinedSigs = info.attributesStr + " " + info.methodsStr; // Combinar atributos y métodos
        std::regex typeRegex(R"(\b([A-Z][a-zA-Z0-9_]+)\b)"); // Asumiendo que los nombres de clase comienzan con mayúscula

        auto words_begin = std::sregex_iterator(combinedSigs.begin(), combinedSigs.end(), typeRegex); // Buscar tipos en firmas
        auto words_end = std::sregex_iterator(); // Fin de la búsqueda

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) { // Iterar sobre los tipos encontrados
            std::string potentialClass = (*i).str(); // Obtener el nombre del tipo
            std::cout << "Signature analysis in " << sourceClass << ": found type " << potentialClass << std::endl;
            if (classes_.count(potentialClass) && potentialClass != sourceClass) {
                // Determinar si es atributo o parte de un método
                if (info.attributesStr.find(potentialClass) != std::string::npos) {
                     scores[sourceClass][potentialClass] += ATTRIBUTE_SCORE;
                } else {
                     scores[sourceClass][potentialClass] += METHOD_SIG_SCORE;
                }
            }
        }
    }

    void analyzeBodyInstantiations(const std::string& sourceClass, const ClassInfo& info, std::map<std::string, std::map<std::string, int>>& scores) {
         for (const auto& pair : classes_) {
            const std::string& targetClass = pair.first;
            if (sourceClass == targetClass) continue;

            std::regex instantiationRegex(R"(\bnew\s+)" + targetClass + R"(\s*\(.*\))");
            
            auto words_begin = std::sregex_iterator(info.fullContent.begin(), info.fullContent.end(), instantiationRegex);
            auto words_end = std::sregex_iterator();

            for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                scores[sourceClass][targetClass] += INSTANTIATION_SCORE;
            }
        }
    }
    
    void saveResults(const std::filesystem::path& outputPath, const std::map<std::string, std::map<std::string, int>>& scores) {
        std::ofstream outFile(outputPath);
        if (!outFile.is_open()) {
            throw std::runtime_error("Error: No se pudo crear el archivo CSV de salida.");
        }
        
        outFile << "class,relations\n";
        
        for (const auto& sourcePair : scores) {
            outFile << "\"" << sourcePair.first << "\",\"{";
            bool first = true;
            for (const auto& targetPair : sourcePair.second) {
                if (!first) {
                    outFile << ", ";
                }
                outFile << "'" << targetPair.first << "': " << targetPair.second;
                first = false;
            }
            outFile << "}\"\n";
        }
    }
};

int main(int argc, char* argv[]) {
    // La comprobación original de "argc != 2" ha sido eliminada.
    // Esta es la única comprobación que necesitamos.
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <path_al_csv_de_fase1> <path_del_proyecto_java>" << std::endl;
        return 1;
    }

    std::filesystem::path csvPath(argv[1]);
    if (!std::filesystem::exists(csvPath)) {
        std::cerr << "Error: El archivo CSV no existe en la ruta proporcionada." << std::endl;
        return 1;
    }
    
    std::filesystem::path projectPath(argv[2]);
    if (!std::filesystem::exists(projectPath) || !std::filesystem::is_directory(projectPath)) {
        std::cerr << "Error: La ruta del proyecto Java no existe o no es un directorio." << std::endl;
        return 1;
    }

    try {
        std::string outputFilename = "results/" + csvPath.stem().string() + "_relations.csv";
        RelationAnalyzer analyzer(csvPath, projectPath);
        analyzer.analyzeAndSave(outputFilename);
        std::cout << "Análisis de relaciones completado. Resultados guardados en: " << outputFilename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}