import { root, Rectangle, ImageFill, Artboard, Color } from "scenegraph";

import {
  ARTBOARD_HEIGHT,
  ARTBOARD_PADDING,
  ARTBOARD_WIDTH,
  DOCUMENT_CENTER,
} from "./constants";

import UI_ELEMENTS from "./ui-elements";

export function getArtboardLocation(i) {
  if (i < 4) {
    return [
      DOCUMENT_CENTER[0] + (ARTBOARD_WIDTH + ARTBOARD_PADDING) * i,
      DOCUMENT_CENTER[1],
    ];
  }

  return [
    DOCUMENT_CENTER[0] + (ARTBOARD_WIDTH + ARTBOARD_PADDING) * (i - 4),
    DOCUMENT_CENTER[1] + ARTBOARD_HEIGHT + ARTBOARD_PADDING,
  ];
}

export function addElement(image, element, artboard) {
  const { name } = element;
  const { x, y } = element.position;
  const { width, height } = element.dimension;

  const imageFill = new ImageFill(image);
  const imageWidth = imageFill.naturalWidth;
  const imageHeight = imageFill.naturalHeight;

  const scale = Math.min(width / imageWidth, height / imageHeight);

  const rectNode = new Rectangle();
  rectNode.name = name;
  rectNode.width = imageWidth * scale;
  rectNode.height = imageHeight * scale;
  rectNode.fill = imageFill;
  artboard.addChild(rectNode);
  rectNode.moveInParentCoordinates(x, y);
}

export function generateArtboard(uiPatternName, uiScreen, i) {
  const [x, y] = getArtboardLocation(i);

  const artboard = new Artboard();
  artboard.name = `${uiPatternName}_wireframe_${uiScreen.id}`;
  artboard.width = uiScreen.width;
  artboard.height = uiScreen.height;
  artboard.fill = new Color("#F2F2F2");
  artboard.dynamicLayout = true;

  root.addChild(artboard);
  artboard.moveInParentCoordinates(x, y);

  uiScreen.objects.forEach((element) => {
    const { name } = element;

    if (name === "other") {
      return;
    }

    const image = UI_ELEMENTS[name];

    addElement(image, element, artboard);
  });
}
