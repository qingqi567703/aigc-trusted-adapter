"""Generate reproducible scientific figures from actual course logs."""
import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"work/mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np

COLORS={"baseline":"#666666","augmented":"#377eb8","premask":"#e69f00","full":"#009e73",
        "no_pair":"#984ea3","fixed_gate":"#56b4e9"}
LABELS={"baseline":"Original adapter","augmented":"Augmented","premask":"Input premask",
        "full":"Full method","no_pair":"No paired loss","fixed_gate":"Fixed feature gate"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42,"savefig.facecolor":"white"})

def architecture(output):
    fig,ax=plt.subplots(figsize=(9.4,3.4))
    ax.set(xlim=(0,10),ylim=(0,4))
    ax.axis("off")
    def box(x,y,w,h,text,train=False):
        ax.add_patch(Rectangle((x,y),w,h,facecolor="#e7f3ec" if train else "#f2f2f2",
                               edgecolor="#333333",linewidth=.8))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9)
    def arrow(a,b,color="#555555"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,color=color,linewidth=1))
    box(.0,2.65,1.5,.65,"Caption p")
    box(2,2.65,1.6,.65,"Text encoder\nFrozen")
    box(4.1,2.65,1.7,.65,"VAE posterior\nCached")
    box(6.1,2.65,1.6,.65,"Noised latent\nShared t and noise")
    box(.0,1.25,1.5,.8,"Two corrupted\nedge conditions")
    box(2,1.25,1.6,.8,"Input premask\nR times condition")
    box(4.1,1.25,1.7,.8,"Adapter\n77.00M parameters",True)
    box(6.1,1.25,1.6,.8,"Multiscale gates\n+56.5K parameters",True)
    box(8.1,1.7,1.7,1.15,"Denoising U-Net\nFrozen weights\nInput gradients retained")
    box(.0,.0,1.5,.65,"Given reliability R")
    box(4.1,.0,3.6,.65,"Noise MSE + 0.1 paired consistency")
    arrow((1.5,2.975),(2,2.975))
    ax.plot([2.8,2.8,8.95],[3.3,3.7,3.7],color="#555555",linewidth=1)
    arrow((8.95,3.7),(8.95,2.85))
    arrow((5.8,2.975),(6.1,2.975));arrow((7.7,2.975),(8.1,2.7))
    arrow((1.5,1.65),(2,1.65));arrow((3.6,1.65),(4.1,1.65))
    arrow((5.8,1.65),(6.1,1.65));arrow((7.7,1.65),(8.1,1.9))
    arrow((1.5,.325),(2.8,1.25));arrow((1.5,.325),(6.9,1.25))
    arrow((8.95,1.7),(7.7,.325));arrow((5.0,.65),(5.0,1.25),"#009e73")
    arrow((7.3,.65),(7.3,1.25),"#009e73")
    ax.text(8.95,.4,"One condition at inference",ha="center",fontsize=8)
    fig.tight_layout(pad=.3)
    for extension in ("png","pdf"):
        fig.savefig(output/f"method.{extension}",dpi=280,bbox_inches="tight")
    plt.close(fig)
    formulas={
      "gate":r"$F_l=A_\theta(R\odot c)_l,\quad r_l=\mathrm{Resize}(R),\quad \widetilde{F}_l=F_l\odot r_l\,[1+0.5\tanh g_\phi(F_l,r_l)]$",
      "pair":r"$\mathcal{L}_{\mathrm{pair}}=\frac{1}{B}\sum_i\frac{\|M_i\odot(\hat{\epsilon}_i^{(1)}-\hat{\epsilon}_i^{(2)})\|_2^2}{\max(C\sum M_i,1)},\quad \mathcal{L}=\mathcal{L}_{\mathrm{diff}}+0.1\mathcal{L}_{\mathrm{pair}}$"}
    for name,formula in formulas.items():
        fig=plt.figure(figsize=(9,.65))
        fig.text(.5,.5,formula,ha="center",va="center",fontsize=13)
        fig.savefig(output/f"equation_{name}.png",dpi=300,bbox_inches="tight",pad_inches=.08)
        plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",default="runs/course")
    parser.add_argument("--output",default="outputs/figures")
    parser.add_argument("--architecture-only",action="store_true")
    args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    architecture(output)
    if args.architecture_only:return
    runs=ROOT/args.runs
    data={}
    for folder in sorted(runs.glob("*")):
        if not (folder/"generated/metrics.json").exists():continue
        config=json.loads((folder/"config.json").read_text())
        metrics=json.loads((folder/"generated/metrics.json").read_text())
        rows=[json.loads(line) for line in (folder/"train.jsonl").read_text().splitlines()]
        data[(config["variant"],config["seed"])]=(metrics,rows)
    if len(data)!=10:raise ValueError(f"Expected 10 evaluated runs, found {len(data)}")
    fig,axes=plt.subplots(1,2,figsize=(9,3.0))
    for variant in ("baseline","augmented","premask","full"):
        curves=[]
        for seed in (42,43):
            values=np.array([row["diffusion_loss"] for row in data[(variant,seed)][1]])
            curves.append(np.convolve(values,np.ones(50)/50,mode="valid"))
        mean=np.stack(curves).mean(0)
        axes[0].plot(np.arange(50,len(mean)+50),mean,color=COLORS[variant],label=LABELS[variant],linewidth=1.2)
        metrics=[data[(variant,seed)][0] for seed in (42,43)]
        x=np.array([m["fake_follow_lift"] for m in metrics])
        y=np.array([m["anchor_f1"] for m in metrics])
        axes[1].errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),
                         fmt="o",color=COLORS[variant],capsize=3,label=LABELS[variant])
        offset={"baseline":(-96,-3),"augmented":(6,8),"premask":(8,-21),"full":(8,18)}[variant]
        axes[1].annotate(LABELS[variant],(x.mean(),y.mean()),xytext=offset,
                         textcoords="offset points",fontsize=8,
                         arrowprops={"arrowstyle":"-","color":COLORS[variant],"lw":.6})
    axes[0].set(xlabel="Optimizer steps",ylabel="Noise MSE (50-step mean)")
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(xlabel="False-edge following lift (lower)",ylabel="Trusted-edge F1 (higher)")
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output/"training_and_tradeoff.png",dpi=300)
    fig.savefig(output/"training_and_tradeoff.pdf")
    plt.close(fig)
    result={}
    for key,(metrics,rows) in data.items():
        result[f"{key[0]}_s{key[1]}"]={"warm_step_median_seconds":float(np.median([r["seconds"] for r in rows if r["step"]>10])),
                                      "peak_vram_MiB":max(r["peak_vram_MiB"] for r in rows)}
    (output/"figure_input_timing.json").write_text(json.dumps(result,indent=2))

if __name__=="__main__":main()
