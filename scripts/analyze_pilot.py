"""Make tables and publication-style figures from the registered pilot artefacts."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summary(values):
    a=np.asarray(values,dtype=float)
    return float(a.mean()),float(a.std(ddof=1))


def main():
    root=Path('runs/E001'); output=Path('reports'); figures=output/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    registration=json.loads((root/'registration.json').read_text())
    cfg=registration['config']
    results={}
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            results[arm,seed]=json.loads((root/f'{arm}_s{seed}'/'result.json').read_text())
    def nll(arm,seed,t): return results[arm,seed]['evaluation']['metrics'][str(t)]['nll']
    rows=[]
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            r=results[arm,seed]
            for t in cfg['eval_loops']:
                rows.append(dict(arm=arm,seed=seed,train_loops=r['train_loops'],eval_loops=t,nll=nll(arm,seed,t),
                    tokens_seen=r['tokens_seen'],training_seconds=r['training_seconds'],parameters=r['parameters']))
    with (output/'E001_depth_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    effects={
        'relative - add @4':[nll('relative',s,4)-nll('add',s,4) for s in cfg['seeds']],
        'rotation - add @4':[nll('depth_rope',s,4)-nll('add',s,4) for s in cfg['seeds']],
        'combined - add @4':[nll('combined',s,4)-nll('add',s,4) for s in cfg['seeds']],
        'factorial interaction @4':[nll('combined',s,4)-nll('relative',s,4)-nll('depth_rope',s,4)+nll('add',s,4) for s in cfg['seeds']],
        'normalized skip - relative @4':[nll('normalized_skip',s,4)-nll('relative',s,4) for s in cfg['seeds']],
        'relative vs add extrapolation gain':[(nll('relative',s,4)-nll('relative',s,8))-(nll('add',s,4)-nll('add',s,8)) for s in cfg['seeds']],
    }
    (output/'E001_effects.json').write_text(json.dumps({k:dict(values=v,mean=summary(v)[0],sample_sd=summary(v)[1]) for k,v in effects.items()},indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    colors=dict(zip(cfg['arms'],plt.get_cmap('tab10').colors))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for arm in cfg['arms']:
        matrix=np.array([[nll(arm,s,t) for t in cfg['eval_loops']] for s in cfg['seeds']])
        mean,sd=matrix.mean(0),matrix.std(0,ddof=1)
        axes[0,0].plot(cfg['eval_loops'],mean,marker='o',label=arm,color=colors[arm])
        axes[0,0].fill_between(cfg['eval_loops'],mean-sd,mean+sd,color=colors[arm],alpha=.1)
        curves=[]
        for seed in cfg['seeds']:
            log=[json.loads(line) for line in (root/f'{arm}_s{seed}'/'learning_curve.jsonl').read_text().splitlines()]
            log={r['step']:r for r in log}  # Resume can repeat a logged checkpoint.
            curves.append([log[t]['validation_nll'] for t in [64,128,192]])
        axes[0,1].plot([32768,65536,98304],np.mean(curves,axis=0),marker='o',label=arm,color=colors[arm])
        diag=np.array([[d['update_cosine_previous'] for d in results[arm,s]['evaluation']['diagnostics_first_batch'][1:]] for s in cfg['seeds']])
        axes[1,0].plot(range(2,17),diag.mean(0),label=arm,color=colors[arm])
        times=[results[arm,s]['training_seconds'] for s in cfg['seeds']]
        native=[nll(arm,s,results[arm,s]['train_loops']) for s in cfg['seeds']]
        axes[1,1].errorbar(np.mean(times),np.mean(native),yerr=np.std(native,ddof=1),fmt='o',color=colors[arm],capsize=3)
        label_offset = -10 if arm=='normalized_skip' else (9 if arm=='relative' else 4)
        axes[1,1].annotate(arm,(np.mean(times),np.mean(native)),xytext=(5,label_offset),textcoords='offset points',fontsize=8,color=colors[arm])
    axes[0,0].axvline(4,ls='--',color='gray',lw=1)
    axes[0,0].set(title='One checkpoint, varying evaluation depth',xlabel='Loops',ylabel='Validation NLL per byte',xticks=cfg['eval_loops'])
    axes[0,0].legend(fontsize=8,ncol=2)
    axes[0,1].set(title='Learning at training depth (shallow: T=1)',xlabel='Presented target bytes',ylabel='Validation NLL per byte')
    axes[0,1].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[1,0].set(title='Update alignment; fixed diagnostic batch',xlabel='Loop',ylabel='Cosine with previous update')
    axes[1,1].set(title='Observed training time; NLL at training depth',xlabel='Training seconds (CPU, excludes evaluation)',ylabel='Validation NLL per byte')
    axes[1,1].set_xlim(left=1.0,right=6.8)
    fig.suptitle('E001 · 90,561 parameters · 3 seeds · 98,304 target bytes/run\nExploratory FineWeb byte pilot; bands/error bars = sample SD',fontsize=13)
    fig.savefig(figures/'E001_pilot.png'); fig.savefig(figures/'E001_pilot.svg'); plt.close(fig)
    null=json.loads((output/'E000_null_control.json').read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(9,3),layout='constrained')
    axes[0].plot([x['loop'] for x in null],[x['nll'] for x in null]); axes[0].set(title='Readout invariant to nuisance rotation',xlabel='Loop',ylabel='NLL')
    axes[1].plot([x['loop'] for x in null],[x['state_step_l2'] for x in null]); axes[1].set(title='State keeps moving',xlabel='Loop',ylabel='Mean state displacement')
    fig.savefig(figures/'E000_null_control.png'); plt.close(fig)
    total=sum(r['tokens_seen'] for r in results.values()); seconds=sum(r['training_seconds'] for r in results.values())
    text=['# E001: результаты собственного CPU-пилота','',
        'Статус: исследовательский pilot на FineWeb; не финальная оценка задания.', '',
        '[Протокол до запуска](../research/protocols/E001_cpu_pilot.md) · [Регистрация и хеши](../runs/E001/registration.json) · [CSV всех seed и глубин](E001_depth_metrics.csv) · [Парные эффекты](E001_effects.json)', '',
        f'Выполнено {len(results)} запусков, {total:,} предъявленных target-байтов суммарно. Во всех моделях 90,561 параметр; training T=4, кроме отдельно обученного shallow T=1. Сумма измеренного времени обучения: {seconds:.1f} s (без evaluation; это не GPU-прогноз).', '',
        'Данные: 400 документов, split 322/45/33; validation — 64 документно чередуемых окна, 4096 целевых байтов. Test не использован. 98,304 target-байта на run, 3 seed, общий LR. Метрика NLL/byte; byte-PPL нельзя сравнивать с PPL BPE-моделей.', '',
        '## Наблюдения', '', '| Вариант | NLL@1, mean ± SD | NLL@4, mean ± SD | NLL@8, mean ± SD | Выигрыш 4→8, mean ± SD |',
        '| --- | --- | --- | --- | --- |']
    for arm in cfg['arms']:
        values=[summary([nll(arm,s,t) for s in cfg['seeds']]) for t in [1,4,8]]
        gain=summary([nll(arm,s,4)-nll(arm,s,8) for s in cfg['seeds']])
        text.append('| '+arm+' | '+' | '.join(f'{a:.4f} ± {b:.4f}' for a,b in values+[gain])+' |')
    text+=['','Положительный выигрыш 4→8 означает пользу дополнительных циклов у тех же весов; отрицательный — ухудшение. Shallow@4/@8 является экстраполяцией модели, обученной при T=1.','',
        '## Парные эффекты по seed','','| Сравнение | Значения для seed 17 / 29 / 43 | Mean ± SD |','| --- | --- | --- |']
    for name,values in effects.items():
        mean,sd=summary(values)
        text.append(f'| {name} | '+ ' / '.join(f'{x:+.5f}' for x in values)+f' | {mean:+.5f} ± {sd:.5f} |')
    text+=['','Для строк NLL@4 отрицательный эффект означает снижение NLL; знак factorial interaction относится к взаимодействию компонентов. В последней строке положительное значение означает, что relative лучше additive сохраняет качество при 4→8. Это описательные оценки при n=3, не заявления о статистической значимости.','',
        '## Графики','','![Качество, обучение, диагностика и время](figures/E001_pilot.png)','',
        '## Ограничения','','Серия короткая, контекст 64, byte vocabulary, небольшая удобная выборка и один LR. Близкие дубликаты не исключены полным промышленным pipeline. Seeds сравниваются на одних документах; ни глубины, ни документы не увеличивают число независимых обучений. Нормировка может менять эффективный LR; следующее сравнение должно это проверить.','',
        'Интерпретация гипотез и решение о следующем эксперименте записываются отдельно в журнале после анализа чисел; исходный протокол сохраняется неизменным.']
    (output/'E001_report.md').write_text('\n'.join(text)+'\n')
    print(f'Analysed {len(results)} runs, {total} target bytes, {seconds:.1f}s training. Reports and figures saved.')


if __name__=='__main__': main()
