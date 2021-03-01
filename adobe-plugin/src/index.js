import { AKIN_URL } from "./constants";
import { generateArtboard } from "./generateArtboard";

// eslint-disable-next-line no-undef
const { error } = require("./lib/dialogs");

const generateScreen = (uiPattern) => async () => {
  try {
    const response = await fetch(AKIN_URL + uiPattern, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Server overloaded`);
    }
    const uiScreens = await response.json();

    uiScreens.forEach((uiScreen, i) => {
      generateArtboard(uiPattern, uiScreen, i);
    });
  } catch (err) {
    await error("Server failed", `Sorry! ${error.message}`);
  }
};

export default {
  commands: {
    generateLoginScreen: generateScreen("login"),
    generateAccountCreationScreen: generateScreen("account_creation"),
    generateProductListing: generateScreen("product_listing"),
    generateProductDescription: generateScreen("product_description"),
    generateSplashScreen: generateScreen("splash"),
  },
};
