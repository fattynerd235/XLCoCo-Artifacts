package com.company;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import JavaParser.JavaParser;
import JavaParser.JavaLexer;
import JavaParser.JavaParserBaseVisitor;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import java.io.IOException;

public class Main  extends JavaParserBaseVisitor {

    public static void main(String[] args) throws IOException {
	// write your code here

        //CharStream input = CharStreams.fromFileName("C:\\Users\\kawser\\IdeaProjects\\CLCDSA\\Source Codes\\AtCoder\\abc001\\A\\4738062.java");
        CharStream input = CharStreams.fromString(
                """
                        import java.util.*;
                        class GFG {
                            static void minChocolates(int A[], int N) {
                                int[] B = new int[N];
                                for (int i = 0; i < N; i++) {
                                    B[i] = 1;
                                }
                                for (int i = 1; i < N; i++) {
                                    if (A[i] > A[i - 1]) B[i] = B[i - 1] + 1;
                                    else B[i] = 1;
                                }
                                for (int i = N - 2; i >= 0; i--) {
                                    if (A[i] > A[i + 1]) B[i] = Math.max(B[i + 1] + 1, B[i]);
                                    else B[i] = Math.max(B[i], 1);
                                }
                                int sum = 0;
                                for (int i = 0; i < N; i++) {
                                    sum += B[i];
                                }
                                System.out.print(sum );
                            }
                            public static void main(String[] args) {
                                int A[] = {23, 14, 15, 14, 56, 29, 14};
                                int N = A.length;
                                minChocolates(A, N);
                            }
                        }
                           
                        """
        );
        JavaLexer lexer = new JavaLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        JavaParser parser = new JavaParser(tokens); // Parser Created
        ParseTree tree = parser.compilationUnit();

        ParseTreeWalker walker = new ParseTreeWalker();
        Listener listener = new Listener(parser);
        walker.walk(listener, tree);

        System.out.println(listener.getVariableno()); // Number of Variables
        System.out.println(listener.getNoofarguments()); // Number of Arguments
        System.out.println(listener.getNoofoperators()); // No of Operators
        System.out.println(listener.getNoofexpression()); // No of Expressions
        System.out.println(listener.getNoofloops()); // No of Loops
        System.out.println(listener.getNoofoperands()); // No of Operands
        System.out.println(listener.getNoofexceptions()); // No of Exceptions
        System.out.println(listener.getNoofexceptionclause()); // No of Exception Handled
        System.out.println(listener.getMccabecomplex()); // Mccabe Complexity
    }
}
