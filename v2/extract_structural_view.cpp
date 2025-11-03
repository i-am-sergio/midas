#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <sstream>

class JavaCodeExtractor {
public:
    JavaCodeExtractor(const std::filesystem::path& inputPath, const std::filesystem::path& outputPath)
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
        outputFile_ << "filename,class,attributes,methods\n";
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

        std::vector<std::string> attributes = findAttributes(content);
        std::vector<std::string> methods = findMethods(content);
        
        writeToCsv(filePath.string(), className, attributes, methods);
    }

    std::string removeComments(std::string code) {
        code = std::regex_replace(code, std::regex("/\\*[\\s\\S]*?\\*/"), "");
        code = std::regex_replace(code, std::regex("//.*"), "");
        return code;
    }

    std::string findEntityName(const std::string& content) {
        std::smatch match;
        std::regex entityRegex(R"(\b(?:class|interface)\s+([a-zA-Z0-9_]+))");
        if (std::regex_search(content, match, entityRegex) && match.size() > 1) {
            return match[1].str();
        }
        return "";
    }

    std::vector<std::string> findAttributes(const std::string& content) {
        std::vector<std::string> attributes;
        std::regex attributeRegex(R"((?:private|public|protected|static|final|\s)*\s*([a-zA-Z0-9_<>\[\]\.]+)\s+([a-zA-Z0-9_]+)\s*(?:=\s*[^;]*)?;)");
        
        size_t classBodyStart = content.find('{');
        if (classBodyStart == std::string::npos) return attributes;

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

            if (line.find('(') == std::string::npos && line.find(')') == std::string::npos) {
                 attributes.push_back(match[1].str() + " " + match[2].str());
            }
        }
        return attributes;
    }

    std::vector<std::string> findMethods(const std::string& content) {
        std::vector<std::string> methods;
        std::regex methodRegex(R"((?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*([a-zA-Z0-9_<>\[\].]+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,.]*)?\s*(?:\{|;))");
        
        auto words_begin = std::sregex_iterator(content.begin(), content.end(), methodRegex);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            methods.push_back(match[1].str() + " " + match[2].str());
        }
        return methods;
    }

    void writeToCsv(const std::string& filename, const std::string& className, const std::vector<std::string>& attributes, const std::vector<std::string>& methods) {
        outputFile_ << "\"" << filename << "\",\"" << className << "\",\"[";
        
        for (size_t i = 0; i < attributes.size(); ++i) {
            outputFile_ << attributes[i] << (i == attributes.size() - 1 ? "" : "; ");
        }
        
        outputFile_ << "]\",\"[";
        
        for (size_t i = 0; i < methods.size(); ++i) {
            outputFile_ << methods[i] << (i == methods.size() - 1 ? "" : "; ");
        }
        
        outputFile_ << "]\"\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <path_del_proyecto_java>" << std::endl;
        return 1;
    }

    std::filesystem::path inputPath(argv[1]);
    if (!std::filesystem::exists(inputPath) || !std::filesystem::is_directory(inputPath)) {
        std::cerr << "Error: La ruta proporcionada no existe o no es un directorio." << std::endl;
        return 1;
    }
    
    std::string projectName = inputPath.filename().string();
    std::string outputFilename = "results/"+projectName + "_fase1.csv";

    try {
        JavaCodeExtractor extractor(inputPath, outputFilename);
        extractor.run();
        std::cout << "Proceso completado. Resultados guardados en: " << outputFilename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}