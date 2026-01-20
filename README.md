# Floor Matching System 🏠 -> 🪵

Calculates which flooring products look most like the floor in your room photo.

## What is this?
This tool helps you find flooring products that match a photo of a room.
1. You give it a **Query Image** (a photo of a room).
2. You select the floor area in that photo.
3. The system compares it against a database of **Product Images** (close-ups of wood/tile).
4. It tells you which products match best!

## 🛠️ How to Inteall

1. **Install Python** (If you haven't already).
2. **Open a terminal** in this folder.
3. **Install the required libraries** by running this command:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

1. **Add Your Images**:
   - Put your **Room Photos** inside: `data/query/`
   - Put your **Product Photos** inside: `data/sku/`

2. **Start the Program**:
   Run this command in your terminal:
   ```bash
   python3 main.py
   ```

3. **Select the Floor**:
   - A window will pop up showing your room image.
   - **Click and drag** a specific box around a clear part of the floor.
   - Press **SPACE** or **ENTER** to confirm.
   - (If the selection is bad, press **c** to cancel and skip that image).

4. **See Results**:
   - The system will think for a moment...
   - Then it will show you the **Best Match** side-by-side!
   - It also prints the top 5 matches in the terminal.

## 🧠 How it Works (The Logic)

The system uses a smart **Two-Step Process** to find the best match:

### Step 1: The "Color Check" 🎨
First, it acts like a fast filter. It looks at the **Color** of your selected floor and compares it to all products.
- It finds products with similar colors (browns, greys, warm tones, etc.).
- It keeps the top candidates and discards the ones that differ completely.
- **Why?** This makes the system fast and accurate.

### Step 2: The "Detail Check" 🔍
Next, it takes those top color candidates and looks closer using **Artificial Intelligence (AI)** (specifically `LightGlue` with `SuperPoint`/`DISK` and now **`LoFTR`**).
- It looks for matching **Patterns, Textures, and Key Features** (like wood grain knots or tile lines).
- It gives a "Feature Score" based on how many tiny details match.

### Final Score
The system combines these two checks to give a final ranking:
- **Color Score** (25% importance)
- **Feature/Detail Score** (75% importance)

## 📁 Folder Structure

- `main.py`: The brain of the operation. Run this!
- `data/`: Where your images live.
- `src/`: The engine room containing code for features, matching, and settings.
