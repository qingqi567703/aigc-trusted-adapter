"""Build the course report and presentation from reviewed, real result files."""
import argparse,html,json
from pathlib import Path
from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle,Image,Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
LABEL={"baseline":"原适配器","augmented":"同增强基线","premask":"输入预掩码","full":"完整方法",
       "no_pair":"去除配对项","fixed_gate":"固定门控"}

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--content",default="docs/report_sections.json")
    parser.add_argument("--analysis",default="docs/results/course_analysis.json")
    parser.add_argument("--interpretation",default="docs/results/interpretation.json")
    parser.add_argument("--figures",default="docs/figures")
    parser.add_argument("--runs",default="runs/course")
    parser.add_argument("--output",default="outputs/课程实验报告.pdf")
    parser.add_argument("--slides",default="outputs/课堂汇报.pdf")
    parser.add_argument("--font",default="C:/Windows/Fonts/simsun.ttc")
    parser.add_argument("--draft",action="store_true")
    args=parser.parse_args()
    content=read(ROOT/args.content)
    analysis=read(ROOT/args.analysis) if not args.draft else None
    interpretation=read(ROOT/args.interpretation) if not args.draft else {
        "abstract":"[待正式实验验证]","diagnosis":"[待正式实验验证]",
        "cases":"[待正式实验验证]","conclusion":"[待正式实验验证]"}
    if not args.draft:
        assert analysis["protocol"]["n_images"]==400
        for key in ("abstract","diagnosis","cases","conclusion"):
            assert interpretation[key] and not any(marker in interpretation[key] for marker in ("[ACTUAL","[待","待正式实验验证")),key
    elif not all("work" in Path(value).parts for value in (args.output,args.slides)):
        raise ValueError("Draft output must remain under work")
    pdfmetrics.registerFont(TTFont("Chinese",args.font,subfontIndex=0))
    bold=Path(args.font).with_name("simhei.ttf")
    pdfmetrics.registerFont(TTFont("ChineseBold",str(bold if bold.exists() else Path(args.font)),subfontIndex=0))
    pdfmetrics.registerFontFamily("Chinese",normal="Chinese",bold="ChineseBold",
                                 italic="Chinese",boldItalic="ChineseBold")
    body=ParagraphStyle("body",fontName="Chinese",fontSize=11,leading=16.5,
                        firstLineIndent=22,spaceAfter=7,wordWrap="CJK")
    caption=ParagraphStyle("caption",fontName="Chinese",fontSize=9,leading=12.5,
                           spaceAfter=8,wordWrap="CJK",textColor=colors.HexColor("#444444"))
    small=ParagraphStyle("small",fontName="Chinese",fontSize=9.5,leading=13.5,wordWrap="CJK")
    heading=ParagraphStyle("heading",fontName="ChineseBold",fontSize=16,leading=22,spaceAfter=12)
    title=ParagraphStyle("title",fontName="ChineseBold",fontSize=20,leading=27,spaceAfter=6)
    subtitle=ParagraphStyle("subtitle",fontName="Chinese",fontSize=11,leading=16,spaceAfter=12)
    W,H=A4;M=48;WIDTH=W-2*M
    figures=ROOT/args.figures
    runs=ROOT/args.runs
    def p(text,style=body):
        return Paragraph(html.escape(text).replace("\n","<br/>"),style)
    def fmt(item,scale=1,decimals=3):
        if item is None:return "待验证"
        return f"{item['mean']*scale:.{decimals}f}±{item['seed_std']*scale:.{decimals}f}"
    def table(rows,widths):
        wrapped=[[p(str(cell),small) for cell in row] for row in rows]
        value=Table(wrapped,colWidths=widths)
        value.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8edf2")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#fafafa")]),
            ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#d9d9d9")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6)]))
        return value
    def make_table(kind):
        if kind=="protocol":
            rows=[["项目","冻结配置"],["训练","8,000 张，1,500 步，全局 batch 8，micro batch 4"],
                  ["优化","AdamW，学习率 1e-5，权重衰减 0.01，梯度裁剪 1"],
                  ["精度与保存","BF16，冻结 U-Net，梯度检查点，每 500 步保存"],
                  ["重复与消融","四主配置各两个训练种子；两个消融各一个种子"],
                  ["测试与生成","前 400 张冻结测试图，每图三个视图，DDIM 25 步"]]
            return table(rows,[85,WIDTH-85])
        if kind=="main_results":
            rows=[["方法","边缘 F1 ↑","错误跟随 ↓","CLIP ↑","KID×1000 ↓","FID400 ↓"]]
            for variant in ("baseline","augmented","premask","full"):
                m=analysis["main"][variant] if analysis else None
                rows.append([LABEL[variant]]+([fmt(m[k],1000 if k=="kid_clean" else 1,2 if k=="fid_clean" else 3 if k=="kid_clean" else 4)
                             for k in ("anchor_f1","fake_follow_lift","clip_cosine","kid_clean","fid_clean")] if m else ["待验证"]*5))
            return table(rows,[83]+[(WIDTH-83)/5]*5)
        if kind=="ablation":
            rows=[["种子 42 配置","边缘 F1 ↑","错误跟随 ↓","CLIP ↑"]]
            for variant in ("full","no_pair","fixed_gate"):
                m=analysis["ablation_seed42"][variant]["metrics"] if analysis else None
                rows.append([LABEL[variant]]+([f"{m[k]:.4f}" for k in ("anchor_f1","fake_follow_lift","clip_cosine")] if m else ["待验证"]*3))
            return table(rows,[143]+[(WIDTH-143)/3]*3)
        if kind=="confidence":
            rows=[["R 系数","图像数","预掩码跟随","完整方法跟随","F1 差值"]]
            groups=analysis["strata_descriptive"]["low_trust"] if analysis else {str(v):None for v in (0,.15,.35,.55)}
            for key,group in groups.items():
                if group:
                    a,b=group["full"],group["premask"]
                    values=[str(a["n_images"])] + [f"{item['fake_follow_lift']:.4f}" if item["fake_follow_lift"] is not None else "无新增笔画" for item in (b,a)]
                    values.append(f"{a['anchor_f1']-b['anchor_f1']:+.4f}")
                else:values=["待验证"]*4
                rows.append([key]+values)
            return table(rows,[60,65,125,125,WIDTH-375])
        values=list(analysis["runs"].values()) if analysis else []
        mean_wall=sum(x["status"]["wall_seconds"] for x in values)/len(values)/60 if values else None
        peak=max(x["status"]["peak_vram_MiB"] for x in values)/1024 if values else None
        return table([["项目","实际记录或协议"],["硬件","4×RTX 4090，24GB 每卡；PyTorch 2.2.2 / CUDA 12.1"],
                      ["每次训练",f"平均 {mean_wall:.2f} 分钟，1,500 步" if mean_wall else "待正式统计"],
                      ["训练峰值",f"最大 {peak:.2f} GiB" if peak else "待正式统计"],
                      ["规模","10 次训练；400 图×3 视图×10 次生成"],
                      ["统计口径","训练计时为日志中的循环耗时；部署与平台待机另计"]],[90,WIDTH-90])
    def picture(name,max_height):
        path=figures/name
        if not path.exists():
            if not args.draft:raise FileNotFoundError(path)
            value=Spacer(1,max_height)
            value.image_placeholder=True
            return value
        with PILImage.open(path) as source:iw,ih=source.size
        scale=min(WIDTH/iw,max_height/ih)
        return Image(str(path),width=iw*scale,height=ih*scale)
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    doc=canvas.Canvas(str(output),pagesize=A4)
    doc.setTitle(content["title"]);doc.setAuthor("生成式人工智能导论课程项目")
    all_chunks=[]
    for index,page in enumerate(content["pages"],1):
        items=[]
        if index==1:
            items += [p(content["title"],title),p(content["subtitle"],subtitle),
                      p("摘要  "+interpretation["abstract"],small),Spacer(1,12)]
        items.append(p(page["heading"],heading))
        texts=page["paragraphs"]
        items.extend(p(text) for text in texts)
        if "equations" in page:
            items.append(picture(page["equations"][0],65))
            items.append(Spacer(1,8))
            items.append(p(page["after_equations"][0]))
            items.append(picture(page["equations"][1],65))
            items.append(Spacer(1,8))
            items.extend(p(text) for text in page["after_equations"][1:])
        if "table" in page:
            items += [Spacer(1,5),make_table(page["table"]),Spacer(1,10)]
        for text in page.get("after_table",[]):
            text=text.replace("[ACTUAL_DIAGNOSIS]",interpretation["diagnosis"]).replace("[ACTUAL_CONCLUSION]",interpretation["conclusion"])
            items.append(p(text))
        if "diagnostic_table" in page:
            items += [Spacer(1,5),make_table(page["diagnostic_table"]),Spacer(1,6),
                      p("可信度分组为描述性诊断；F1 差值为完整方法减预掩码。R 是控制系数，不是校准概率。",caption)]
        if "figure" in page:
            items += [picture(page["figure"],370 if index==7 else 175),Spacer(1,6),p(page["caption"],caption)]
        for text in page.get("after_figure",[]):
            items.append(p(text.replace("[ACTUAL_CASE_ANALYSIS]",interpretation["cases"])))
        cursor=0
        if all_chunks:all_chunks.append([Spacer(1,12)])
        while cursor<len(items):
            item=items[cursor]
            if isinstance(item,Paragraph) and item.style.name=="heading" and cursor+1<len(items):
                all_chunks.append(items[cursor:cursor+2]);cursor+=2
            elif (isinstance(item,Image) or getattr(item,"image_placeholder",False)) and cursor+2<len(items) and isinstance(items[cursor+2],Paragraph) and items[cursor+2].style.name=="caption":
                all_chunks.append(items[cursor:cursor+3]);cursor+=3
            else:
                all_chunks.append([item]);cursor+=1
    pages=[[]];used=0
    capacity=H-M-45
    for chunk in all_chunks:
        height=sum(item.wrap(WIDTH,capacity)[1]+item.getSpaceAfter() for item in chunk)
        if height>capacity:raise ValueError("An indivisible report block exceeds page height")
        if used+height>capacity:
            pages.append([]);used=0
        if not pages[-1] and all(isinstance(item,Spacer) for item in chunk):continue
        pages[-1].append(chunk);used+=height
    if not args.draft and not 6<=len(pages)<=8:
        raise ValueError(f"Body must contain 6-8 naturally paginated pages, found {len(pages)}")
    for index,chunks in enumerate(pages,1):
        y=H-M
        for chunk in chunks:
            for item in chunk:
                width,height=item.wrap(WIDTH,y-45)
                left=M+(WIDTH-width)/2 if isinstance(item,Image) else M
                item.drawOn(doc,left,y-height)
                y-=height+item.getSpaceAfter()
        doc.setFont("Chinese",9)
        doc.drawString(M,24,"生成式人工智能导论 期末实验报告")
        doc.drawRightString(W-M,24,f"正文 {index} / {len(pages)}")
        if args.draft:
            doc.setFont("ChineseBold",10)
            doc.drawRightString(W-M,H-20,"排版草稿 正式结果待验证")
        doc.showPage()
    y=H-M
    head=p("参考文献",heading);_,h=head.wrap(WIDTH,700);head.drawOn(doc,M,y-h);y-=h+14
    for index,text in enumerate(content["references"],1):
        value=p(f"[{index}] {text}",small);_,h=value.wrap(WIDTH,y-45)
        if h>y-45:doc.showPage();y=H-M
        value.drawOn(doc,M,y-h);y-=h+12
    doc.setFont("Chinese",9);doc.drawRightString(W-M,24,"参考文献 不计入正文");doc.showPage()
    doc.save()
    build_slides(ROOT/args.slides,content,interpretation,analysis,runs,figures,args.draft,pdfmetrics)
    print(json.dumps({"report":str(output),"body_pages":len(pages),"references_pages":1,
                      "slides":str(ROOT/args.slides),"draft":args.draft},ensure_ascii=False))

def build_slides(destination,content,interpretation,analysis,runs,figures,draft,pdfmetrics):
    SW,SH=960,540;M=48
    destination.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(destination),pagesize=(SW,SH))
    c.setTitle("局部可信度条件图像生成 课堂汇报")
    normal=ParagraphStyle("slide",fontName="Chinese",fontSize=19,leading=29,wordWrap="CJK")
    small=ParagraphStyle("slide_small",fontName="Chinese",fontSize=14,leading=22,wordWrap="CJK")
    slide_number=0
    def begin(title):
        nonlocal slide_number
        slide_number+=1
        c.setFillColor(colors.black);c.setFont("ChineseBold",28)
        c.drawString(M,SH-55,title)
        c.setFont("Chinese",11);c.drawRightString(SW-M,20,str(slide_number))
        if draft:c.drawString(M,20,"排版草稿 正式结果待验证")
    def text(value,x,y,width=864,style=normal):
        para=Paragraph(html.escape(value),style)
        _,height=para.wrap(width,500);para.drawOn(c,x,y-height)
        return y-height-18
    def image(path,x,y,width,max_height=360):
        if not path.exists():
            return text("图像待实验完成",x,y,width,small)
        with PILImage.open(path) as source:iw,ih=source.size
        height=width*ih/iw
        if height>max_height:width*=max_height/height;height=max_height
        c.drawImage(str(path),x,y-height,width=width,height=height,preserveAspectRatio=True,mask="auto")
        return y-height
    slides=[
      ("问题在哪里",["错误笔画可能被生成模型解释为真实结构。","降低整张控制图的强度，会同时削弱可靠轮廓。","问题设定：已给定局部可信度，能否减少错误复制并保持可信布局？"]),
      ("主论文与实际训练",["T2I-Adapter，AAAI 2024 主轨长文，CCF A 类。","共同初始化 SD1.5 与作者 sketch Adapter；原基线和改进均实际微调。","约 77M Adapter 参数更新，完整方法新增 56.5K 门控参数。"]),
      ("模型与梯度路径",[]),
      ("我们的局部修改",["输入条件乘 R；在四个尺度用 R 与特征预测有限调整。","同照片两种干预共享文本、latent、时刻与噪声，在可信区域约束预测一致。","空间控制强度已有先行研究；重点是验证相对简单预掩码的附加价值。"]),
      ("冻结实验协议",["8,000 训练图，400 开发图；首轮冻结测试 400 图。","四主方法各两个训练种子，加两个同种子消融，共 10 次训练。","每次 1,500 步，全局 batch 8；测试每图三个视图、25 步、相同输入噪声。","R 是给定的合成控制系数；真实自由手绘和自动可靠性估计未验证。"]),
      ("主实验结果",[]),
      ("简单基线与消融",[]),
      ("相同输入生成对比",[]),
      ("继续检查固定案例",[]),
      ("证据边界",["报告全部结果，并单列共同排除安全过滤图像的敏感性分析。","两个训练种子与图像 bootstrap 反映不同的不确定性。","FID-400 是小样本辅助指标，不能与 FID-50K 或公开完整训练排名比较。","单骨干、合成边缘与 oracle 区域 R 限制了结论的适用范围。"]),
      ("结论与可复现材料",[interpretation.get("slide_conclusion",interpretation["conclusion"]),"保留训练配置、日志、检查点、数据与模型快照、逐种子结果和配对统计。","课程级正向结果不自动构成独立顶会贡献。"])]
    for title,bullets in slides:
        begin(title)
        y=440
        if title=="模型与梯度路径":
            y=image(figures/"method.png",M,y,864,330)
            text("冻结权重仍保留输入梯度；VAE 后验缓存后继续采样；推理只需一份条件。",M,y-18,864,small)
        elif title=="主实验结果":
            c.setFont("ChineseBold",18)
            columns=[M,M+235,M+440,M+650]
            for x,label in zip(columns,["方法","可信边缘 F1 ↑","错误跟随 ↓","CLIP cosine ↑"]):c.drawString(x,420,label)
            for row,variant in enumerate(("baseline","augmented","premask","full")):
                y=360-row*60;c.setFont("Chinese",18);c.drawString(M,y,LABEL[variant])
                for x,metric in zip(columns[1:],("anchor_f1","fake_follow_lift","clip_cosine")):
                    value=analysis["main"][variant][metric] if analysis else None
                    c.drawString(x,y,f"{value['mean']:.4f} ± {value['seed_std']:.4f}" if value else "待验证")
            text("均值 ± 两个训练种子标准差；错误跟随只统计确有新增错误笔画的样例。",M,115,864,small)
        elif title=="相同输入生成对比":
            y=image(figures/"formal_slides_first2.png",M,y,864,335)
            text("冻结测试清单第 1、2 张。多余钟面与黑色小物体展示了未解决的幻觉。",M,y-15,864,small)
        elif title=="继续检查固定案例":
            y=image(figures/"formal_slides_next2.png",M,y,864,335)
            text("同一清单第 3、4 张。冲浪场景的错误结构减轻，但预掩码也能解释大部分改善。",M,y-15,864,small)
        elif title=="简单基线与消融":
            columns=[M,M+235,M+440,M+650]
            c.setFont("ChineseBold",18)
            for x,label in zip(columns,["种子 42 配置","可信边缘 F1 ↑","错误跟随 ↓","CLIP cosine ↑"]):c.drawString(x,420,label)
            for row,variant in enumerate(("full","no_pair","fixed_gate")):
                y=360-row*55;c.setFont("Chinese",18);c.drawString(M,y,LABEL[variant])
                m=analysis["ablation_seed42"][variant]["metrics"] if analysis else None
                for x,metric in zip(columns[1:],("anchor_f1","fake_follow_lift","clip_cosine")):
                    c.drawString(x,y,f"{m[metric]:.4f}" if m else "待验证")
            y=text("同一训练种子对照；共享 R 的输入预掩码是主实验中的关键强基线。",M,195,864,small)
            text(interpretation.get("slide_diagnosis",interpretation["diagnosis"]),M,y,864,small)
        else:
            for bullet in bullets:y=text(bullet,M,y)
            if y<40:raise ValueError(f"Slide {slide_number} overflows; shorten reviewed text")
        c.showPage()
    c.save()

if __name__=="__main__":main()
