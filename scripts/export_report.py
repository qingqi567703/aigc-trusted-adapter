"""Export the same real report content as editable Markdown."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LABEL={'baseline':'原适配器','augmented':'同增强基线','premask':'输入预掩码','full':'完整方法',
       'no_pair':'去除配对项','fixed_gate':'固定门控'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='docs/REPORT.md')
    args=parser.parse_args()
    read=lambda name:json.loads((ROOT/name).read_text(encoding='utf-8'))
    content=read('docs/report_sections.json');a=read('docs/results/course_analysis.json')
    interp=read('docs/results/interpretation.json')
    resolve=lambda value:value.replace('[ACTUAL_DIAGNOSIS]',interp['diagnosis']).replace('[ACTUAL_CONCLUSION]',interp['conclusion']).replace('[ACTUAL_CASE_ANALYSIS]',interp['cases'])
    lines=['# '+content['title'],content['subtitle'],'**摘要** '+interp['abstract']]
    def table(rows):
        return '\n'.join(['| '+' | '.join(map(str,rows[0]))+' |',
                          '|'+'---|'*len(rows[0])]+['| '+' | '.join(map(str,row))+' |' for row in rows[1:]])
    def block(kind):
        if kind=='main_results':
            rows=[['方法','可信边缘 F1 ↑','错误跟随 ↓','CLIP ↑','KID×1000 ↓','FID-400 ↓']]
            for v in ('baseline','augmented','premask','full'):
                row=[LABEL[v]]
                for k in ('anchor_f1','fake_follow_lift','clip_cosine','kid_clean','fid_clean'):
                    m=a['main'][v][k];s=1000 if k=='kid_clean' else 1
                    row.append(f"{m['mean']*s:.4f} ± {m['seed_std']*s:.4f}")
                rows.append(row)
            return table(rows)
        if kind=='ablation':
            rows=[['配置（种子 42）','可信边缘 F1 ↑','错误跟随 ↓','CLIP ↑']]
            for v in ('full','no_pair','fixed_gate'):
                m=a['ablation_seed42'][v]['metrics']
                rows.append([LABEL[v]]+[f'{m[k]:.4f}' for k in ('anchor_f1','fake_follow_lift','clip_cosine')])
            return table(rows)
        if kind=='protocol':
            return table([['项目','冻结配置'],['训练','8,000 图，1,500 步，全局 batch 8，micro batch 4'],
                          ['优化','AdamW，lr 1e-5，weight decay 0.01，梯度裁剪 1'],
                          ['精度','BF16，U-Net 冻结并保留输入梯度，梯度检查点'],
                          ['生成','400 图，每图 3 视图，DDIM 25 步，CFG 7.5']])
        if kind=='confidence':
            rows=[['R','图像数','预掩码跟随','完整方法跟随','F1 差值']]
            for key,g in a['strata_descriptive']['low_trust'].items():
                b,f=g['premask'],g['full']
                rows.append([key,f['n_images'],f"{b['fake_follow_lift']:.4f}",f"{f['fake_follow_lift']:.4f}",f"{f['anchor_f1']-b['anchor_f1']:+.4f}"])
            return table(rows)
        return table([['项目','实测'],['硬件','4×RTX 4090 24GB；Torch 2.2.2 / CUDA 12.1'],
                      ['每次训练','平均 16.35 分钟，最大已分配显存 9.97 GiB'],
                      ['总流程','训练阶段约 49.9 分钟；训练、生成、评测约 108.7 分钟']])
    for page in content['pages']:
        lines.append('## '+page['heading'])
        lines.extend(page['paragraphs'])
        if 'equations' in page:
            lines.append('![特征门控公式](figures/'+page['equations'][0]+')')
            lines.append(page['after_equations'][0])
            lines.append('![配对训练公式](figures/'+page['equations'][1]+')')
            lines.extend(page['after_equations'][1:])
        if 'table' in page:lines.append(block(page['table']))
        lines.extend(resolve(t) for t in page.get('after_table',[]))
        if 'diagnostic_table' in page:lines.append(block(page['diagnostic_table']))
        if 'figure' in page:
            lines.extend(['!['+page['caption']+'](figures/'+page['figure']+')',page['caption']])
        lines.extend(resolve(t) for t in page.get('after_figure',[]))
    lines+=['## 参考文献']+[f'[{i}] {text}' for i,text in enumerate(content['references'],1)]
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print(output)

if __name__=='__main__':main()
