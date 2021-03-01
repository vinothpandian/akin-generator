import image from "@rollup/plugin-image";

export default {
  input: "src/index.js",
  output: {
    file: "main.js",
    format: "cjs",
    exports: "default",
  },
  plugins: [image()],
  external: ["scenegraph", "viewport"],
};
