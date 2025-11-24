/**
 * Extracts the first valid JSON object from a string.
 * Useful for parsing LLM outputs that may contain markdown or other text.
 */
export function extractJSON(text: string): any | null {
  try {
    // First try parsing the whole string
    return JSON.parse(text);
  } catch (e) {
    // If that fails, look for JSON-like structure
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        return JSON.parse(jsonMatch[0]);
      } catch (e) {
        // If simple match fails, try to find the first balanced brace pair
        // This is a simple implementation and might need to be more robust
        let braceCount = 0;
        let startIndex = -1;
        
        for (let i = 0; i < text.length; i++) {
          if (text[i] === '{') {
            if (braceCount === 0) startIndex = i;
            braceCount++;
          } else if (text[i] === '}') {
            braceCount--;
            if (braceCount === 0 && startIndex !== -1) {
              try {
                const potentialJson = text.substring(startIndex, i + 1);
                return JSON.parse(potentialJson);
              } catch (e) {
                // Continue searching if this block wasn't valid JSON
                startIndex = -1;
              }
            }
          }
        }
      }
    }
    return null;
  }
}
