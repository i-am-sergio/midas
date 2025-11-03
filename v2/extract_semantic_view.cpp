#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <sstream>

class SemanticCollector {
public:
    SemanticCollector(const std::filesystem::path& inputPath, const std::filesystem::path& outputPath)
        : inputDirectory_(inputPath) {
        outputFile_.open(outputPath);
        if (!outputFile_.is_open()) {
            throw std::runtime_error("Error: No se pudo abrir el archivo de salida.");
        }
    }

    void run() {
        writeCsvHeader();
        processDirectory(inputDirectory_);
    }

private:
    std::filesystem::path inputDirectory_;
    std::ofstream outputFile_;

    void writeCsvHeader() {
        outputFile_ << "class_name,concatenated_text\n";
    }

    void processDirectory(const std::filesystem::path& path) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".java") {
                parseFile(entry.path());
            }
        }
    }

    void parseFile(const std::filesystem::path& filePath) {
        std::ifstream fileStream(filePath);
        if (!fileStream.is_open()) {
            std::cerr << "Advertencia: No se pudo abrir el archivo " << filePath << std::endl;
            return;
        }

        std::stringstream buffer;
        buffer << fileStream.rdbuf();
        std::string content = buffer.str();
        
        content = removeComments(content);

        std::string className = findEntityName(content);
        if (className.empty()) {
            return;
        }
        
        std::string packageName = findPackageName(content);
        std::vector<std::string> attributeNames = findAttributeNames(content);
        std::vector<std::string> methodNames = findMethodNames(content);

        std::string concatenatedText = buildConcatenatedString(packageName, className, attributeNames, methodNames);
        
        writeToCsv(className, concatenatedText);
    }

    std::string removeComments(std::string code) {
        code = std::regex_replace(code, std::regex("/\\*[\\s\\S]*?\\*/"), "");
        code = std::regex_replace(code, std::regex("//.*"), "");
        return code;
    }

    std::string findPackageName(const std::string& content) {
        std::smatch match;
        std::regex packageRegex(R"(package\s+([\w.]+);)");
        if (std::regex_search(content, match, packageRegex) && match.size() > 1) {
            return match[1].str();
        }
        return "";
    }

    std::string findEntityName(const std::string& content) {
        std::smatch match;
        std::regex entityRegex(R"(\b(?:class|interface)\s+([a-zA-Z0-9_]+))");
        if (std::regex_search(content, match, entityRegex) && match.size() > 1) {
            return match[1].str();
        }
        return "";
    }

    std::vector<std::string> findAttributeNames(const std::string& content) {
        std::vector<std::string> names;
        // Captura el tipo (grupo 1) y el nombre (grupo 2) de un atributo
        std::regex attributeRegex(R"((?:private|public|protected|static|final|\s)*\s*([a-zA-Z0-9_<>\[\]\.]+)\s+([a-zA-Z0-9_]+)\s*(?:=\s*[^;]*)?;)");
        
        size_t classBodyStart = content.find('{');
        if (classBodyStart == std::string::npos) return names;
        std::string classBody = content.substr(classBodyStart);

        auto words_begin = std::sregex_iterator(classBody.begin(), classBody.end(), attributeRegex);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            std::string line = match.prefix().str();
            
            size_t lastNewline = line.rfind('\n');
            if(lastNewline != std::string::npos) {
                line = line.substr(lastNewline + 1);
            }
            // Heurística simple: si la línea no contiene paréntesis, es probable que sea un atributo y no un método
            if (line.find('(') == std::string::npos && line.find(')') == std::string::npos) {
                 names.push_back(match[2].str()); // Captura solo el nombre
            }
        }
        return names;
    }

    std::vector<std::string> findMethodNames(const std::string& content) {
        std::vector<std::string> names;
        // Captura el tipo de retorno (grupo 1) y el nombre del método (grupo 2)
        std::regex methodRegex(R"((?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*([a-zA-Z0-9_<>\[\].]+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,.]*)?\s*(?:\{|;))");
        
        auto words_begin = std::sregex_iterator(content.begin(), content.end(), methodRegex);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            names.push_back(match[2].str()); // Captura solo el nombre
        }
        return names;
    }
    
    std::string buildConcatenatedString(const std::string& packageName, const std::string& className, const std::vector<std::string>& attributes, const std::vector<std::string>& methods) {
        std::stringstream ss;
        ss << packageName << " " << className;
        for(const auto& attr : attributes) {
            ss << " " << attr;
        }
        for(const auto& method : methods) {
            ss << " " << method;
        }
        return ss.str();
    }

    void writeToCsv(const std::string& className, const std::string& concatenatedText) {
        outputFile_ << "\"" << className << "\",\"" << concatenatedText << "\"\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <path_del_proyecto_java>" << std::endl;
        return 1;
    }

    std::filesystem::path inputPath(argv[1]);
    std::filesystem::path canonicalPath; 

    try {
        canonicalPath = std::filesystem::canonical(inputPath);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: La ruta proporcionada no es válida o no existe: " << e.what() << std::endl;
        return 1;
    }
    
    if (!std::filesystem::is_directory(canonicalPath)) {
        std::cerr << "Error: La ruta proporcionada no es un directorio." << std::endl;
        return 1;
    }
    
    std::string projectName = canonicalPath.filename().string();
    std::string outputFilename = "results/"+projectName + "_fase1_semantic_view.csv";

    try {
        SemanticCollector collector(canonicalPath, outputFilename); 
        collector.run();
        std::cout << "Proceso completado. Vista semántica guardada en: " << outputFilename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}