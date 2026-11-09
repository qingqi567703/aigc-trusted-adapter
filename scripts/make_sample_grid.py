"""Create fixed-order report figures from real saved test outputs."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",default="runs/course")
    parser.add_argument("--output",default="docs/figures")
    args=parser.parse_args()
    runs=ROOT/args.runs;output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (ROOT/"docs/resources/splits/test.jsonl").read_text(encoding="utf-8").splitlines()][:4]
    font_path=next((path for path in (Path("C:/Windows/Fonts/arial.ttf"),Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) if path.exists()),None)
    font=ImageFont.truetype(str(font_path),58) if font_path else ImageFont.load_default(size=58)
    tile,header,gap=512,112,8
    labels=("Reference","Condition","Reliability","Original","Premask","Full")
    def build(selected,name):
        sheet=Image.new("RGB",(tile*6,header+len(selected)*(tile+gap)),"white")
        draw=ImageDraw.Draw(sheet)
        for column,label in enumerate(labels):
            draw.text((column*tile+20,24),label,font=font,fill="black")
        for index,row in enumerate(selected):
            key=row["id"]
            source=runs/"full_s42/generated"
            paths=[source/f"references/{key}.png",source/f"conditions/{key}_0.png",source/f"conditions/{key}_trust.png",
                   runs/f"baseline_s42/generated/samples/{key}_0.png",
                   runs/f"premask_s42/generated/samples/{key}_0.png",runs/f"full_s42/generated/samples/{key}_0.png"]
            for column,path in enumerate(paths):
                with Image.open(path) as image:
                    sheet.paste(image.convert("RGB").resize((tile,tile),Image.Resampling.LANCZOS),(column*tile,header+index*(tile+gap)))
        sheet.save(output/name)
    build(rows,"formal_first4.png")
    build(rows[:2],"formal_slides_first2.png")
    build(rows[2:],"formal_slides_next2.png")
    (output/"figure_cases.json").write_text(json.dumps({"rule":"First four frozen test-manifest entries, no result-dependent selection","ids":[row["id"] for row in rows],
                                                       "training_seed":42,"view":"0"},indent=2),encoding="utf-8")
    print("Created fixed test grids", [row["id"] for row in rows])

if __name__=="__main__":main()
