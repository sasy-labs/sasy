import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: ["dist/", "src/generated/", "node_modules/"],
  },
  ...tseslint.configs.strict,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Allow unused vars prefixed with _
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // Allow empty functions (reset stubs etc.)
      "@typescript-eslint/no-empty-function": "off",
      // Allow non-null assertions — gRPC callbacks guarantee non-null on success
      "@typescript-eslint/no-non-null-assertion": "off",
    },
  },
);
