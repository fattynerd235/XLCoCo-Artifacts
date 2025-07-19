package com.company;

import CSharpParser.CSharpLexer;
import CSharpParser.CSharpParser;
import JavaParser.JavaLexer;
import JavaParser.JavaParser;
import PythonParser.PythonLexer;
import PythonParser.PythonParser;
import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvValidationException;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class XLCostFeatureCollection {
    static String contentDir = "./Source Codes/XLCost";
    public static void main(String[] args){
        List<Path> allfiles = getAllFiles(contentDir);
        allfiles.forEach( fileName -> {
            /*
            if (fileName.toString().contains("Java_C#_nonclone"))
                java_csharp_feature_collection(fileName);

            else */ if(fileName.toString().contains("Java_Python_nonclone"))
                java_python_feature_collection(fileName);
            else if(fileName.toString().contains("C#_Python_nonclone"))
                csharp_python_feature_collection(fileName);

            }
        );
    }

    private static List<Path> getAllFiles(String fileName){
        Path dirPath = Paths.get(fileName);
        List<Path> fileList = null;
        try (Stream<Path> stream = Files.list(dirPath)) {
            // Collect all regular files into a List
            fileList = stream
                    .filter(Files::isRegularFile)
                    .collect(Collectors.toList());
        } catch (IOException e) {
            System.err.println("Error reading directory: " + e.getMessage());
        }
        return fileList;
    }

    private static void java_csharp_feature_collection(Path filename) {
        try(CSVReader reader = new CSVReader(new FileReader(filename.toAbsolutePath().toString()))){
            String fileName = filename.getFileName().toString();
            // Remove extension
            int dotIndex = fileName.lastIndexOf('.');
            String nameWithoutExt = (dotIndex == -1) ? fileName : fileName.substring(0, dotIndex);

            CSVWriter writer =new CSVWriter(new FileWriter(contentDir + "/" + nameWithoutExt + "features.csv"));
            String[] header = {"src_source_code", "src_feature1", "src_feature2", "src_feature3", "src_feature4",
                    "src_feature5", "src_feature6", "src_feature7", "src_feature8", "src_feature9", "des_source_code",
                    "des_feature1", "des_feature2", "des_feature3", "des_feature4", "des_feature5", "des_feature6",
                    "des_feature7", "des_feature8", "des_feature9"};
            writer.writeNext(header);

            String[] line;
            String[] features = new String[20];
            reader.readNext(); // skip header line
            while((line = reader.readNext()) != null){

                if(line.length>=2){
                    int count = -1;
                    //features[count] = line[0];
                    String[] subfeatures;

                    String sourceCode = line[0];
                    features[++count] = sourceCode;
                    subfeatures = JavaCollection(sourceCode);
                    for(int i = 0; i<subfeatures.length;i++)
                        features[++count] = subfeatures[i];

                    String desCode = line[1];
                    features[++count] = desCode;
                    subfeatures = CSharpCollection(desCode);
                    for(int i=0; i<subfeatures.length; i++)
                        features[++count] = subfeatures[i];

                    writer.writeNext(features);
                }
                else{
                    System.out.println("Error..... the line is " + line.toString());
                }
            }
        }catch (IOException ex){
            System.out.println(ex.getMessage());
            ex.printStackTrace();
        } catch (CsvValidationException e) {
            e.printStackTrace();
        }
    }

    private static void java_python_feature_collection(Path filename) {
        try(CSVReader reader = new CSVReader(new FileReader(filename.toAbsolutePath().toString()))){
            String fileName = filename.getFileName().toString();
            // Remove extension
            int dotIndex = fileName.lastIndexOf('.');
            String nameWithoutExt = (dotIndex == -1) ? fileName : fileName.substring(0, dotIndex);

            CSVWriter writer =new CSVWriter(new FileWriter(contentDir + "/" + nameWithoutExt + "features.csv"));
            String[] header = {"src_source_code", "src_feature1", "src_feature2", "src_feature3", "src_feature4",
                    "src_feature5", "src_feature6", "src_feature7", "src_feature8", "src_feature9", "des_source_code",
                    "des_feature1", "des_feature2", "des_feature3", "des_feature4", "des_feature5", "des_feature6",
                    "des_feature7", "des_feature8", "des_feature9"};
            writer.writeNext(header);

            String[] line;
            String[] features = new String[20];
            reader.readNext(); // skip header line
            while((line = reader.readNext()) != null){
                if (line.length>=2){
                    int count = -1;
                    //features[count] = line[0];
                    String[] subfeatures;

                    String sourceCode = line[0];
                    features[++count] = sourceCode;
                    subfeatures = JavaCollection(sourceCode);
                    for(int i = 0; i<subfeatures.length;i++)
                        features[++count] = subfeatures[i];

                    String desCode = line[1];
                    features[++count] = desCode;
                    subfeatures = PythonCollection(desCode);
                    for(int i=0; i<subfeatures.length; i++)
                        features[++count] = subfeatures[i];

                    writer.writeNext(features);
                }
                else{
                    System.out.println("Error..... the line is " + line.toString());
                }
            }
        }catch (IOException ex){
            System.out.println(ex.getMessage());
            ex.printStackTrace();
        } catch (CsvValidationException e) {
            e.printStackTrace();
        }
    }

    private static void csharp_python_feature_collection(Path filename) {
        try(CSVReader reader = new CSVReader(new FileReader(filename.toAbsolutePath().toString()))){
            String fileName = filename.getFileName().toString();
            // Remove extension
            int dotIndex = fileName.lastIndexOf('.');
            String nameWithoutExt = (dotIndex == -1) ? fileName : fileName.substring(0, dotIndex);

            CSVWriter writer =new CSVWriter(new FileWriter(contentDir + "/" + nameWithoutExt + "features.csv"));
            String[] header = {"src_source_code", "src_feature1", "src_feature2", "src_feature3", "src_feature4",
                    "src_feature5", "src_feature6", "src_feature7", "src_feature8", "src_feature9", "des_source_code",
                    "des_feature1", "des_feature2", "des_feature3", "des_feature4", "des_feature5", "des_feature6",
                    "des_feature7", "des_feature8", "des_feature9"};
            writer.writeNext(header);

            String[] line;
            String[] features = new String[20];
            reader.readNext(); // skip header line
            while((line = reader.readNext()) != null) {
                if (line.length >= 2) {
                    int count = -1;
                    //features[count] = line[0];
                    String[] subfeatures;

                    String sourceCode = line[0];
                    features[++count] = sourceCode;
                    subfeatures = CSharpCollection(sourceCode);
                    for (int i = 0; i < subfeatures.length; i++)
                        features[++count] = subfeatures[i];

                    String desCode = line[1];
                    features[++count] = desCode;
                    subfeatures = PythonCollection(desCode);
                    for (int i = 0; i < subfeatures.length; i++)
                        features[++count] = subfeatures[i];

                    writer.writeNext(features);
                } else {
                    System.out.println("Error..... the line is " + line.toString());
                }
            }
        }catch (IOException ex){
            System.out.println(ex.getMessage());
            ex.printStackTrace();
        } catch (CsvValidationException e) {
            e.printStackTrace();
        }
    }

    private static String[] JavaCollection(String sourceCode){
        System.out.println("Working with Java Codes");
        CharStream input = CharStreams.fromString(sourceCode);
        JavaLexer lexer = new JavaLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        JavaParser parser = new JavaParser(tokens); // Parser Created
        ParseTree tree = parser.compilationUnit();

        ParseTreeWalker walker = new ParseTreeWalker();
        Listener listener = new Listener(parser);
        walker.walk(listener, tree);
        String[] features = new String[9];
        features[0] = Integer.toString(listener.getVariableno());
        features[1] = Integer.toString(listener.getNoofarguments());
        features[2] = Integer.toString(listener.getNoofoperators());
        features[3] = Integer.toString(listener.getNoofexpression());
        features[4] = Integer.toString(listener.getNoofloops());
        features[5] = Integer.toString(listener.getNoofoperands());
        features[6] = Integer.toString(listener.getNoofexceptions());
        features[7] = Integer.toString(listener.getNoofexceptionclause());
        features[8] = Integer.toString(listener.getMccabecomplex());

        return features;
    }

    private static String[] CSharpCollection(String desCode){
        System.out.println("Working with CSharp Codes");
        String[] features = new String[9];
        /// Parsing Code ///
        CharStream input = CharStreams.fromString(desCode);
        CSharpLexer lexer = new CSharpLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CSharpParser parser = new CSharpParser(tokens);
        ParseTree tree = parser.compilation_unit();

        ParseTreeWalker walker = new ParseTreeWalker();
        CSharpListener listener = new CSharpListener(parser);
        walker.walk(listener, tree);

        features[0] = Integer.toString(listener.getNoofvariables());
        features[1] = Integer.toString(listener.getNoofArguments());
        features[2] = Integer.toString(listener.getNoofOperators());
        features[3] = Integer.toString(listener.getNoofExpressions());
        features[4] = Integer.toString(listener.getNoofLoops());
        features[5] = Integer.toString(listener.getNoofOperands());
        features[6] = Integer.toString(listener.getNoofExceptions());
        features[7] = Integer.toString(listener.getNoofHandledExceptions());
        features[8] = Integer.toString(listener.getMccabecomplex());

        return features;
    }

    private static String[] PythonCollection(String sourceCode){
        System.out.println("Working with Python Codes");
        /// Parsing Code ///
        CharStream input = CharStreams.fromString(sourceCode);
        PythonLexer lexer = new PythonLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        PythonParser parser = new PythonParser(tokens);
        ParseTree tree = parser.root();

        ParseTreeWalker walker = new ParseTreeWalker();
        PythonListener listener = new PythonListener(parser);
        walker.walk(listener, tree);
        String[] features = new String[9];
        features[0] = Integer.toString(listener.getNoOfVariables());
        features[1] = Integer.toString(listener.getNoOfArguments());
        features[2] = Integer.toString(listener.getNoOfOperators());
        features[3] = Integer.toString(listener.getNoOfExceptions());
        features[4] = Integer.toString(listener.getNoOfLoops());
        features[5] = Integer.toString(listener.getNoOfOperands());
        features[6] = Integer.toString(listener.getNoOfExceptions());
        features[7] = Integer.toString(listener.getNoOfExceptionClauses());
        features[8] = Integer.toString(listener.getMccabeComplexity());

        return features;
    }
}
