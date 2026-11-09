"""Build an editable anonymous Word report from the verified course results."""
import argparse,json
from copy import deepcopy
from pathlib import Path
from docx import Document
from docx.shared import Pt,Inches,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_BREAK
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
LABEL={'baseline':'原适配器','augmented':'同增强基线','premask':'输入预掩码','full':'完整方法','no_pair':'去除配对项','fixed_gate':'固定门控'}

def node(name,*children,**attributes):
    e=OxmlElement(name)
    for key,value in attributes.items():e.set(qn(key.replace('_',':')),str(value))
    for child in children:
        if isinstance(child,list):
            for item in child:e.append(item)
        else:e.append(child)
    return e

def mr(text):
    t=node('m:t');t.text=text;t.set(qn('xml:space'),'preserve')
    return node('m:r',t)

def sub(base,index):return node('m:sSub',node('m:e',base),node('m:sub',mr(index)))
def acc(base):return node('m:acc',node('m:accPr',node('m:chr',m_val='̂')),node('m:e',base))
def sup_sub(base,index,power):
    return node('m:sSubSup',node('m:e',base),node('m:sub',mr(index)),node('m:sup',mr(power)))
def frac(top,bottom):return node('m:f',node('m:num',top),node('m:den',bottom))
def loss(index):return sub(mr('ℒ'),index)

def equation(doc,parts):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent=Pt(0)
    p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(7)
    p.paragraph_format.line_spacing=1.15
    p._p.append(node('m:oMath',parts))
    return p

def gates(doc):
    equation(doc,[sub(mr('F'),'l'),mr(' = '),sub([sub(mr('A'),'θ'),mr('(R ⊙ c)')],'l'),mr('，  '),sub(mr('r'),'l'),mr(' = Resize(R)')])
    tilde=node('m:acc',node('m:accPr',node('m:chr',m_val='̃')),node('m:e',mr('F')))
    equation(doc,[sub(tilde,'l'),mr(' = '),sub(mr('F'),'l'),mr(' ⊙ '),sub(mr('r'),'l'),mr('[1 + 0.5 tanh '),sub(mr('g'),'φ'),mr('('),sub(mr('F'),'l'),mr(', '),sub(mr('r'),'l'),mr(')]')])

def paired(doc):
    norm=[mr('‖'),sub(mr('M'),'i'),mr(' ⊙ ('),sup_sub(acc(mr('ε')),'i','(1)'),mr(' − '),sup_sub(acc(mr('ε')),'i','(2)'),mr(')‖')]
    numerator=sup_sub(norm,'2','2')
    denominator=[mr('max(C ∑ '),sub(mr('M'),'i'),mr(', 1)')]
    nary=node('m:nary',node('m:naryPr',node('m:chr',m_val='∑'),node('m:limLoc',m_val='subSup'),node('m:supHide',m_val='1')),node('m:sub',mr('i')),node('m:sup'),node('m:e',frac(numerator,denominator)))
    equation(doc,[loss('pair'),mr(' = '),frac(mr('1'),mr('B')),mr(' '),nary])
    equation(doc,[mr('ℒ = '),loss('diff'),mr(' + 0.1 '),loss('pair')])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/匿名实验报告_可编辑.docx');args=p.parse_args()
    read=lambda name:json.loads((ROOT/name).read_text(encoding='utf-8'))
    content=read('docs/report_sections.json');a=read('docs/results/course_analysis.json');interp=read('docs/results/interpretation.json')
    resolve=lambda text:text.replace('[ACTUAL_DIAGNOSIS]',interp['diagnosis']).replace('[ACTUAL_CONCLUSION]',interp['conclusion']).replace('[ACTUAL_CASE_ANALYSIS]',interp['cases'])
    doc=Document();section=doc.sections[0]
    # Preserve the existing report's A4 design rather than applying an unrelated template.
    section.page_width=Pt(595.276);section.page_height=Pt(841.89)
    for name in ('top_margin','bottom_margin','left_margin','right_margin'):setattr(section,name,Pt(48))
    section.header_distance=Pt(20);section.footer_distance=Pt(20)
    usable=section.page_width-section.left_margin-section.right_margin
    normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(11)
    normal.font.color.rgb=RGBColor(0,0,0);normal.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    normal.paragraph_format.line_spacing=Pt(16.5);normal.paragraph_format.space_after=Pt(7)
    normal.paragraph_format.first_line_indent=Pt(22);normal.paragraph_format.widow_control=True
    normal.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.element.get_or_add_pPr().append(node('w:snapToGrid',w_val='0'))
    for name,size,east in [('Title',20,'黑体'),('Subtitle',11,'宋体'),('Heading 1',16,'黑体'),('Caption',9,'宋体')]:
        s=doc.styles[name];s.font.name='Times New Roman';s.font.size=Pt(size);s.font.color.rgb=RGBColor(0,0,0)
        s.font.bold=name in ('Title','Heading 1');s.font.italic=False
        s.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),east)
        s.paragraph_format.first_line_indent=Pt(0);s.paragraph_format.line_spacing=Pt(size*1.35)
        if name=='Heading 1':s.paragraph_format.keep_with_next=True;s.paragraph_format.space_before=Pt(12);s.paragraph_format.space_after=Pt(10)
    doc.styles['Title'].paragraph_format.space_after=Pt(6)
    doc.styles['Subtitle'].paragraph_format.space_after=Pt(10)
    doc.styles['Caption'].paragraph_format.space_after=Pt(8)
    for border in list(doc.styles.element.xpath('.//w:pBdr')):border.getparent().remove(border)
    doc.core_properties.author='';doc.core_properties.last_modified_by=''
    doc.core_properties.title=content['title'];doc.core_properties.subject='生成式人工智能导论期末实验报告'
    doc.core_properties.comments=''
    settings=doc.settings.element
    settings.append(node('w:updateFields',w_val='true'))
    def para(text,style=None):return doc.add_paragraph(resolve(text),style)
    def smallpara(text):
        v=para(text);v.paragraph_format.first_line_indent=Pt(0);v.paragraph_format.line_spacing=Pt(13.5)
        for r in v.runs:r.font.size=Pt(9.5)
        return v
    def picture(name,max_height=175):
        from PIL import Image
        path=ROOT/'docs/figures'/name
        with Image.open(path) as im:w,h=im.size
        width=min(usable,Pt(max_height)*w/h)
        v=doc.add_paragraph();v.alignment=WD_ALIGN_PARAGRAPH.CENTER;v.paragraph_format.first_line_indent=Pt(0)
        v.paragraph_format.line_spacing=1.0
        v.paragraph_format.keep_with_next=True;v.paragraph_format.space_after=Pt(4)
        shape=v.add_run().add_picture(str(path),width=int(width))
        shape._inline.docPr.set('descr',name.replace('_',' '))
    def table(rows,widths):
        t=doc.add_table(rows=0,cols=len(rows[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for col,width in zip(t.columns,widths):col.width=Pt(width)
        pr=t._tbl.tblPr
        borders=node('w:tblBorders')
        for side in ('top','left','bottom','right','insideH','insideV'):borders.append(node('w:'+side,w_val='single',w_sz='4',w_color='D9D9D9'))
        pr.append(borders);pr.append(node('w:tblCellMar',*[node('w:'+side,w_w=str(value),w_type='dxa') for side,value in [('top',100),('bottom',100),('left',100),('right',100)]]))
        for index,data in enumerate(rows):
            row=t.add_row();row._tr.get_or_add_trPr().append(node('w:cantSplit'))
            if index==0:row._tr.get_or_add_trPr().append(node('w:tblHeader'))
            for col,(cell,text,width) in enumerate(zip(row.cells,data,widths)):
                cell.width=Pt(width);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if index==0:cell._tc.get_or_add_tcPr().append(node('w:shd',w_fill='E8EDF2'))
                elif index%2==0:cell._tc.get_or_add_tcPr().append(node('w:shd',w_fill='FAFAFA'))
                v=cell.paragraphs[0];v.paragraph_format.first_line_indent=Pt(0)
                v.paragraph_format.space_after=Pt(0);v.paragraph_format.space_before=Pt(0);v.paragraph_format.line_spacing=Pt(13.5)
                v.alignment=WD_ALIGN_PARAGRAPH.CENTER if col and len(data)>2 else WD_ALIGN_PARAGRAPH.LEFT
                r=v.add_run(str(text));r.font.size=Pt(9.5)
        spacer=doc.add_paragraph();spacer.paragraph_format.space_after=Pt(2);spacer.paragraph_format.line_spacing=Pt(2)
    def make_table(kind):
        W=usable/12700
        if kind=='main_results':
            rows=[['方法','边缘 F1 ↑','错误跟随 ↓','CLIP ↑','KID×1000 ↓','FID400 ↓']]
            for variant in ('baseline','augmented','premask','full'):
                row=[LABEL[variant]]
                for k in ('anchor_f1','fake_follow_lift','clip_cosine','kid_clean','fid_clean'):
                    m=a['main'][variant][k];scale=1000 if k=='kid_clean' else 1;dec=2 if k=='fid_clean' else 3 if k=='kid_clean' else 4
                    row.append(f"{m['mean']*scale:.{dec}f}±{m['seed_std']*scale:.{dec}f}")
                rows.append(row)
            table(rows,[83]+[(W-83)/5]*5)
        elif kind=='ablation':
            rows=[['种子 42 配置','边缘 F1 ↑','错误跟随 ↓','CLIP ↑']]
            for variant in ('full','no_pair','fixed_gate'):
                m=a['ablation_seed42'][variant]['metrics'];rows.append([LABEL[variant]]+[f'{m[k]:.4f}' for k in ('anchor_f1','fake_follow_lift','clip_cosine')])
            table(rows,[143]+[(W-143)/3]*3)
        elif kind=='confidence':
            rows=[['R 系数','图像数','预掩码跟随','完整方法跟随','F1 差值']]
            for key,g in a['strata_descriptive']['low_trust'].items():
                b,f=g['premask'],g['full'];rows.append([key,str(f['n_images']),f"{b['fake_follow_lift']:.4f}",f"{f['fake_follow_lift']:.4f}",f"{f['anchor_f1']-b['anchor_f1']:+.4f}"])
            table(rows,[60,65,125,125,W-375])
        elif kind=='protocol':
            table([['项目','冻结配置'],['训练','8,000 张，1,500 步，全局 batch 8，micro batch 4'],['优化','AdamW，学习率 1e-5，权重衰减 0.01，梯度裁剪 1'],['精度与保存','BF16，冻结 U-Net，梯度检查点，每 500 步保存'],['重复与消融','四主配置各两个训练种子；两个消融各一个种子'],['测试与生成','前 400 张冻结测试图，每图三个视图，DDIM 25 步']],[85,W-85])
        else:
            table([['项目','实际记录或协议'],['硬件','4×RTX 4090，24GB 每卡；PyTorch 2.2.2 / CUDA 12.1'],['每次训练','平均 16.35 分钟，1,500 步'],['训练峰值','最大 9.97 GiB'],['规模','10 次训练；400 图×3 视图×10 次生成'],['统计口径','训练计时为日志中的循环耗时；部署与平台待机另计']],[90,W-90])
    def field(p,instruction):
        r=p.add_run();r._r.append(node('w:fldChar',w_fldCharType='begin'))
        code=node('w:instrText');code.set(qn('xml:space'),'preserve');code.text=' '+instruction+' ';r._r.append(code)
        r._r.append(node('w:fldChar',w_fldCharType='end'))
    footer=section.footer.paragraphs[0];footer.paragraph_format.first_line_indent=Pt(0)
    footer.paragraph_format.tab_stops.add_tab_stop(usable,WD_ALIGN_PARAGRAPH.RIGHT)
    footer.add_run('生成式人工智能导论 期末实验报告\t正文 ');field(footer,'PAGE');footer.add_run(' / ');field(footer,'SECTIONPAGES')
    for r in footer.runs:r.font.size=Pt(9)
    para(content['title'],'Title');para(content['subtitle'],'Subtitle');smallpara('摘要  '+interp['abstract'])
    for index,page in enumerate(content['pages'],1):
        para(page['heading'],'Heading 1')
        for text in page['paragraphs']:para(text)
        if 'equations' in page:
            gates(doc);para(page['after_equations'][0]);paired(doc)
            for text in page['after_equations'][1:]:para(text)
        if 'table' in page:make_table(page['table'])
        for text in page.get('after_table',[]):para(text)
        if 'diagnostic_table' in page:
            make_table(page['diagnostic_table']);para('可信度分组为描述性诊断；F1 差值为完整方法减预掩码。R 是控制系数，不是校准概率。','Caption')
        if 'figure' in page:
            picture(page['figure'],370 if index==7 else 175);para(page['caption'],'Caption')
        for text in page.get('after_figure',[]):para(text)
    refs=doc.add_section(WD_SECTION_START.NEW_PAGE);refs.footer.is_linked_to_previous=False
    footer=refs.footer.paragraphs[0];footer.paragraph_format.first_line_indent=Pt(0);footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('参考文献 不计入正文').font.size=Pt(9)
    para('参考文献','Heading 1')
    for i,text in enumerate(content['references'],1):smallpara(f'[{i}] {text}')
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True);doc.save(output)
    print(json.dumps({'output':str(output),'native_word_equations':4,'native_tables':5,'anonymous':True},ensure_ascii=False))

if __name__=='__main__':main()
