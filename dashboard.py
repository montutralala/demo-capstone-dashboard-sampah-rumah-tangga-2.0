import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
import os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Trashkara Dashboard",
    page_icon="♻️", layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, "main_data.csv"))

df = load_data()
CAT_COLOR = {'Anorganik': '#3498db', 'B3': '#e74c3c', 'Organik': '#2ecc71'}
ORDER_KES  = ['Mudah', 'Sedang', 'Sulit', 'Sangat Sulit']
CATLIST    = ['Anorganik', 'B3', 'Organik']

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/recycle-sign.png", width=80)
st.sidebar.title("♻️ Trashkara")
st.sidebar.markdown("**Dataset Klasifikasi Sampah Indonesia**")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("📌 Navigasi", [
    "🏠 Overview",
    "📊 EDA & Distribusi",
    "🌿 Q1 — Emisi CO₂e",
    "💰 Q2 — Beban Biaya",
    "🔬 Feature Engineering",
    "🧪 A/B Testing",
    "📋 Data Dictionary"
])

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter")
kat_filter = st.sidebar.multiselect(
    "Kategori Sampah",
    options=CATLIST,
    default=CATLIST
)
df_f = df[df['Kategori'].isin(kat_filter)] if kat_filter else df

# ── OVERVIEW ─────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a5276,#2ecc71);
                color:white;padding:22px;border-radius:12px;margin-bottom:18px'>
    <h1 style='margin:0'>♻️ Trashkara — Dashboard Analisis Data Sampah Indonesia</h1>
    <p style='margin:6px 0 0'>47 jenis sampah · 3 kategori · 14.100 sampel · 2 pertanyaan bisnis SMART</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📦 Total Sampel", f"{len(df_f):,}")
    c2.metric("🗂️ Jenis Sampah", f"{df_f['Nama Sampah'].nunique()}")
    c3.metric("☁️ Avg Emisi CO₂e", f"{df_f['Emisi CO2e (kg)'].mean():.3f} kg")
    c4.metric("💸 Avg Cost Burden", f"{df_f['cost_burden'].mean():.2f}")
    c5.metric("⚠️ High-Cost Burden", f"{df_f['high_cost_burden'].mean()*100:.0f}% sampel")

    st.markdown("---")
    cl, cr = st.columns(2)

    with cl:
        st.subheader("🎯 Pertanyaan Bisnis (Framework SMART)")
        st.markdown("""
**❓ Q1 — Emisi CO₂e & Kesulitan Daur Ulang**
> Kategori sampah manakah yang menyumbang rata-rata emisi CO₂e tertinggi, dan apakah tingkat kesulitan daur ulang berkorelasi positif secara signifikan?
*(Diuji dengan Spearman Correlation, α=0.05)*

---
**❓ Q2 — Struktur Beban Biaya Lintas Kategori**
> Bagaimana distribusi beban biaya (nilai jual mentah ke pabrik vs biaya proses industri) lintas kategori, dan jenis sampah mana yang masuk kuadran high-cost burden?

> 📌 **Konteks:** *Nilai Jual* = harga sampah mentah ke pabrik. *Biaya Proses* = biaya operasional industri (mesin, energi, dll.). *Cost Burden* = biaya proses − nilai jual.
        """)

    with cr:
        fig, ax = plt.subplots(figsize=(6,4))
        cat_j = df_f.groupby('Kategori')['Nama Sampah'].nunique()
        ax.pie(cat_j, labels=cat_j.index, autopct='%1.1f%%',
               colors=[CAT_COLOR[k] for k in cat_j.index], startangle=90,
               wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title('Komposisi Jenis Sampah per Kategori')
        st.pyplot(fig); plt.close()

    st.subheader("🗃️ Preview Dataset (20 Baris Pertama)")
    st.dataframe(df_f[['No','Nama Sampah','Kategori','Emisi CO2e (kg)',
                        'cost_burden','high_cost_burden','kategori_urai']].head(20),
                 use_container_width=True)

# ── EDA ───────────────────────────────────────────────────────────────────────
elif page == "📊 EDA & Distribusi":
    st.title("📊 Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi Emisi CO₂e")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(df_f['Emisi CO2e (kg)'].dropna(), bins=25, color='#e74c3c', edgecolor='white', alpha=0.85)
        ax.axvline(df_f['Emisi CO2e (kg)'].mean(), color='#922b21', linestyle='--', lw=2,
                   label=f"Mean={df_f['Emisi CO2e (kg)'].mean():.2f}")
        ax.set_xlabel('Emisi CO₂e (kg)'); ax.set_ylabel('Frekuensi'); ax.legend()
        st.pyplot(fig); plt.close()
        st.caption("💡 Distribusi merata dengan mean ≈2.03 kg. Setiap jenis sampah memiliki emisi tetap (300 baris identik per jenis).")

    with c2:
        st.subheader("Distribusi Cost Burden")
        fig, ax = plt.subplots(figsize=(8,4))
        cb_cnt = df_f['cost_burden'].value_counts().sort_index()
        colors_cb = ['#27ae60' if v<=0 else '#f39c12' if v==1 else '#e74c3c' for v in cb_cnt.index]
        ax.bar(cb_cnt.index.astype(str), cb_cnt.values, color=colors_cb, edgecolor='white')
        ax.set_xlabel('Cost Burden (Biaya Proses − Nilai Jual)'); ax.set_ylabel('Frekuensi')
        for b, v in zip(ax.patches, cb_cnt.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+30, f'{v:,}', ha='center', fontsize=9)
        st.pyplot(fig); plt.close()
        st.caption("💡 Mayoritas sampah memiliki cost_burden positif — industri menanggung defisit operasional.")

    st.subheader("Profil Nilai Jual vs Biaya Proses per Kategori")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8,4))
        ekon = df_f.groupby('Kategori')[['nj_encoded','bp_encoded']].mean()
        x = np.arange(len(ekon)); w = 0.35
        b1 = ax.bar(x-w/2, ekon['nj_encoded'], w, label='Nilai Jual Mentah', color='#27ae60', edgecolor='white')
        b2 = ax.bar(x+w/2, ekon['bp_encoded'], w, label='Biaya Proses Industri', color='#e74c3c', edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(ekon.index); ax.legend(); ax.set_ylabel('Encoded Value (1–4)')
        ax.set_title('Nilai Jual vs Biaya Proses per Kategori')
        for b, v in list(zip(b1, ekon['nj_encoded']))+list(zip(b2, ekon['bp_encoded'])):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f'{v:.2f}', ha='center', fontsize=10)
        st.pyplot(fig); plt.close()
        st.caption("💡 Semua kategori: biaya proses (merah) > nilai jual mentah (hijau). Industri pengolahan sampah secara umum tidak mandiri finansial dari nilai jual mentah saja.")

    with c2:
        fig, ax = plt.subplots(figsize=(8,4))
        cb_cat = df_f.groupby('Kategori')['cost_burden'].mean().sort_values()
        colors_p = ['#27ae60' if v<=0 else '#e74c3c' for v in cb_cat.values]
        bars = ax.barh(cb_cat.index, cb_cat.values, color=colors_p, edgecolor='white', height=0.5)
        ax.axvline(0, color='black', linestyle='--', lw=1.5, alpha=0.7)
        for bar, val in zip(bars, cb_cat.values):
            ax.text(val+(0.05 if val>=0 else -0.15), bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=11, fontweight='bold')
        ax.set_xlabel('Cost Burden'); ax.set_title('Rata-rata Cost Burden per Kategori')
        st.pyplot(fig); plt.close()
        st.caption("💡 B3 memiliki cost_burden tertinggi — biaya khusus penanganan zat berbahaya sangat mahal. Organik terendah karena proses kompos/biogas relatif murah.")

    st.subheader("🔥 Heatmap Korelasi")
    nc = ['Dapat Terurai','Daur Ulang','Nilai Jual','waktu_urai_bulan','kesulitan_encoded',
          'Emisi CO2e (kg)','nj_encoded','bp_encoded','cost_burden']
    corr = df_f[nc].corr()
    fig, ax = plt.subplots(figsize=(12,7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title('Heatmap Korelasi — Variabel Numerik & Encoded')
    st.pyplot(fig); plt.close()
    st.caption("💡 'bp_encoded' dan 'cost_burden' berkorelasi sangat tinggi (0.88) — cost_burden didominasi biaya proses. 'Daur Ulang' berkorelasi positif dengan 'nj_encoded': sampah yang dapat didaur ulang cenderung memiliki nilai jual mentah lebih tinggi ke pabrik.")

# ── Q1 ────────────────────────────────────────────────────────────────────────
elif page == "🌿 Q1 — Emisi CO₂e":
    st.title("🌿 Q1 — Emisi CO₂e & Kesulitan Daur Ulang")
    st.markdown("""
    <div style='background:#eaf4fb;padding:14px;border-left:4px solid #2874a6;border-radius:6px;margin-bottom:16px'>
    <b>Pertanyaan SMART:</b> Kategori sampah manakah dalam dataset Trashkara 2025 yang menyumbang rata-rata emisi CO₂e tertinggi, dan apakah tingkat kesulitan daur ulang berkorelasi positif secara signifikan dengan besarnya emisi yang dihasilkan?
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for i, cat in enumerate(CATLIST):
        if cat in df_f['Kategori'].unique():
            avg = df_f[df_f['Kategori']==cat]['Emisi CO2e (kg)'].mean()
            [c1,c2,c3][i].metric(f"{cat} — Avg Emisi", f"{avg:.3f} kg")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Boxplot Emisi per Kategori")
        fig, ax = plt.subplots(figsize=(8,5))
        ed = [df_f[df_f['Kategori']==k]['Emisi CO2e (kg)'].dropna()
              for k in CATLIST if k in df_f['Kategori'].unique()]
        lb = [k for k in CATLIST if k in df_f['Kategori'].unique()]
        bp2 = ax.boxplot(ed, labels=lb, patch_artist=True, widths=0.5)
        for patch, color in zip(bp2['boxes'], [CAT_COLOR[k] for k in lb]):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, d in enumerate(ed, 1):
            ax.plot(i, d.mean(), 'D', color='black', ms=8, zorder=5)
        ax.set_ylabel('Emisi CO₂e (kg)'); ax.set_title('Distribusi Emisi per Kategori\n(◆ = rata-rata)')
        st.pyplot(fig); plt.close()
        st.caption("💡 Organik memiliki median emisi tertinggi. Dekomposisi anaerobik sampah organik melepas CH₄ (25× lebih kuat dari CO₂), mendorong emisi ekuivalen menjadi lebih tinggi.")

    with c2:
        st.subheader("Emisi per Tingkat Kesulitan (±SE)")
        emisi_kes = df_f.groupby('Kesulitan Daur Ulang')['Emisi CO2e (kg)'].mean().reindex(ORDER_KES).dropna()
        sem_kes   = df_f.groupby('Kesulitan Daur Ulang')['Emisi CO2e (kg)'].sem().reindex(ORDER_KES).dropna()
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(emisi_kes.index, emisi_kes.values,
                      color=['#2ecc71','#f1c40f','#e67e22','#e74c3c'][:len(emisi_kes)],
                      edgecolor='white', yerr=sem_kes.values, capsize=5, width=0.6)
        for bar, val in zip(bars, emisi_kes.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.07,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
        ax.set_title('Avg Emisi CO₂e per Tingkat Kesulitan'); ax.set_ylabel('Emisi CO₂e (kg)')
        ax.set_ylim(0, emisi_kes.max()*1.3)
        st.pyplot(fig); plt.close()
        st.caption("💡 Tren naik dari Mudah → Sangat Sulit menunjukkan pola positif antara kesulitan daur ulang dan emisi. Error bar (SE) yang kecil mengindikasikan estimasi yang stabil.")

    st.subheader("📐 Uji Spearman: Korelasi Emisi vs Kesulitan Daur Ulang")
    if len(df_f['kesulitan_encoded'].dropna()) > 0:
        rho, pval = spearmanr(df_f['kesulitan_encoded'].dropna(), df_f['Emisi CO2e (kg)'].dropna())
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Spearman ρ", f"{rho:.4f}")
        mc2.metric("p-value", f"{pval:.6f}")
        mc3.metric("Signifikan (α=0.05)?", "✅ Ya" if pval < 0.05 else "❌ Tidak")
        if pval < 0.05:
            st.success(f"✅ Korelasi positif **signifikan** — p={pval:.6f} < 0.05. Semakin sulit didaur ulang, emisi semakin besar (ρ={rho:.4f}).")
            st.info("💡 **Interpretasi:** Meski kekuatan korelasi lemah (ρ≈0.13), dengan n=14.100 ukuran efek kecil pun bermakna secara statistik. Banyak faktor lain (komposisi kimia, proses produksi) turut mempengaruhi emisi selain kesulitan daur ulang.")

    st.subheader("🏆 Top 10 Jenis Sampah — Emisi CO₂e Tertinggi")
    top10 = df_f.groupby(['Nama Sampah','Kategori'])['Emisi CO2e (kg)'].mean().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(12,5))
    bar_c = [CAT_COLOR.get(cat,'#95a5a6') for _, cat in top10.index]
    ax.barh(range(len(top10)), top10.values, color=bar_c, edgecolor='white')
    ax.set_yticks(range(len(top10))); ax.set_yticklabels([n[0] for n in top10.index])
    ax.set_xlabel('Rata-rata Emisi CO₂e (kg)'); ax.set_title('Top 10 Jenis Sampah — Emisi Tertinggi')
    for i, val in enumerate(top10.values): ax.text(val+0.02, i, f'{val:.2f}', va='center', fontsize=9)
    ax.legend(handles=[mpatches.Patch(color=v, label=k) for k,v in CAT_COLOR.items()], fontsize=9)
    st.pyplot(fig); plt.close()
    st.caption("💡 Top emisi didominasi sampah Organik (sisa makanan, daun, dll.) dan sebagian Anorganik berat. B3 tidak mendominasi top-10 emisi CO₂e meskipun paling berbahaya dari sisi toksisitas.")

    st.info("📌 **Kesimpulan Q1:** Kategori Organik menyumbang emisi CO₂e rata-rata tertinggi. Uji Spearman mengkonfirmasi korelasi positif signifikan antara kesulitan daur ulang dan emisi. Prioritaskan infrastruktur composting & biogas untuk menekan emisi GRK dari sampah organik.")

# ── Q2 ────────────────────────────────────────────────────────────────────────
elif page == "💰 Q2 — Beban Biaya":
    st.title("💰 Q2 — Struktur Beban Biaya Operasional Industri Sampah")
    st.markdown("""
    <div style='background:#eafaf1;padding:14px;border-left:4px solid #27ae60;border-radius:6px;margin-bottom:8px'>
    <b>Pertanyaan SMART:</b> Bagaimana distribusi struktur beban biaya (nilai jual mentah dan biaya proses) lintas kategori dalam dataset Trashkara 2025, dan jenis sampah mana yang masuk kuadran high-cost burden?
    </div>
    <div style='background:#fef9e7;padding:10px;border-left:4px solid #f39c12;border-radius:6px;margin-bottom:16px;font-size:0.9em'>
    📌 <b>Nilai Jual (Rp/kg)</b> = harga sampah <b>mentah</b> yang dijual ke pabrik/pengepul <i>sebelum</i> diolah.<br>
    📌 <b>Biaya Proses (Rp/kg)</b> = biaya <b>operasional industri</b>: mesin, energi, tenaga kerja, dll.<br>
    📌 <b>Cost Burden</b> = Biaya Proses − Nilai Jual → semakin positif = defisit industri semakin besar.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for i, cat in enumerate(CATLIST):
        if cat in df_f['Kategori'].unique():
            avg_cb = df_f[df_f['Kategori']==cat]['cost_burden'].mean()
            pct_hcb = df_f[df_f['Kategori']==cat]['high_cost_burden'].mean()*100
            [c1,c2,c3][i].metric(f"{cat}", f"Avg CB: {avg_cb:.2f}", f"{pct_hcb:.0f}% high-cost")

    st.subheader("Heatmap: Matriks Nilai Jual × Biaya Proses per Kategori")
    nj_order = ['Rendah (< Rp 500/kg)','Sedang (Rp 500-2.000/kg)',
                'Sedang (Rp 2.000\u20134.000/kg)','Tinggi (> Rp 2.000/kg)','Tinggi (> Rp 5.000/kg)']
    bp_order = ['Rendah (< Rp 500/kg)','Sedang (Rp 500-2.000/kg)','Sedang (Rp 1.500\u20133.000/kg)',
                'Tinggi (> Rp 2.000/kg)','Sangat Tinggi (> Rp 5.000/kg)']
    nj_s = ['Rendah\n<500','Sedang\n500-2k','Sedang\n2k-4k','Tinggi\n>2k','Tinggi\n>5k']
    bp_s = ['Rendah\n<500','Sedang\n500-2k','Sedang\n1.5k-3k','Tinggi\n>2k','Sgt Tinggi\n>5k']
    uniq_j = df_f.drop_duplicates(['Nama Sampah','Kategori'])[
        ['Nama Sampah','Kategori','Nilai Jual (Rp/kg)','Biaya Proses (Rp/kg)']]

    kats_avail = [k for k in CATLIST if k in df_f['Kategori'].unique()]
    fig, axes = plt.subplots(1, len(kats_avail), figsize=(6*len(kats_avail), 6))
    if len(kats_avail) == 1: axes = [axes]
    for ax, cat in zip(axes, kats_avail):
        sub = uniq_j[uniq_j['Kategori']==cat]
        pivot = pd.crosstab(sub['Biaya Proses (Rp/kg)'], sub['Nilai Jual (Rp/kg)'])
        pivot = pivot.reindex(index=[o for o in bp_order if o in pivot.index],
                              columns=[o for o in nj_order if o in pivot.columns], fill_value=0)
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks(range(len(pivot.columns))); ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([nj_s[nj_order.index(c)] for c in pivot.columns], fontsize=8, rotation=30, ha='right')
        ax.set_yticklabels([bp_s[bp_order.index(r)] for r in pivot.index], fontsize=8)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i,j]
                if v > 0:
                    ax.text(j, i, str(v), ha='center', va='center', fontsize=12, fontweight='bold',
                            color='white' if v >= max(pivot.values.max()*0.6, 1) else 'black')
        ax.set_title(f'Kategori: {cat}', fontweight='bold', color=CAT_COLOR[cat], fontsize=12)
        ax.set_xlabel('Nilai Jual Mentah (Rp/kg)', fontsize=9)
        if cat == kats_avail[0]: ax.set_ylabel('Biaya Proses Industri (Rp/kg)', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.suptitle('Matriks Nilai Jual × Biaya Proses\n(angka = jumlah jenis sampah per kombinasi)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.caption("💡 **Anorganik:** Tersebar luas — ada di pojok kiri-atas (nilai rendah, biaya tinggi = hot-spot beban) dan kanan-bawah (nilai tinggi, biaya rendah = paling viable). **B3:** Seragam di biaya 'Sangat Tinggi' — tidak ada jenis B3 yang biaya prosesnya rendah. **Organik:** Semua di zona biaya rendah-sedang dengan nilai jual sedang-tinggi.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Komposisi Struktur Beban Biaya")
        df_f2 = df_f.copy()
        df_f2['cb_label'] = df_f2['cost_burden'].apply(
            lambda x: 'NJ > BP\n(nilai > biaya)' if x<0 else ('Impas\n(NJ=BP)' if x==0 else 'BP > NJ\n(beban tinggi)'))
        cb_comp = df_f2.groupby(['Kategori','cb_label']).size().unstack(fill_value=0)
        for col in ['NJ > BP\n(nilai > biaya)','Impas\n(NJ=BP)','BP > NJ\n(beban tinggi)']:
            if col not in cb_comp.columns: cb_comp[col] = 0
        cb_comp = cb_comp[['NJ > BP\n(nilai > biaya)','Impas\n(NJ=BP)','BP > NJ\n(beban tinggi)']]
        cb_pct = cb_comp.div(cb_comp.sum(axis=1), axis=0)*100
        fig, ax = plt.subplots(figsize=(8,5))
        cb_pct.plot(kind='bar', stacked=True, ax=ax,
                    color=['#27ae60','#f39c12','#e74c3c'], edgecolor='white', width=0.55)
        ax.set_xlabel('Kategori'); ax.set_ylabel('Persentase (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(loc='upper right', fontsize=9, title='Struktur Beban')
        for patch in ax.patches:
            h = patch.get_height()
            if h > 4:
                ax.text(patch.get_x()+patch.get_width()/2, patch.get_y()+h/2,
                        f'{h:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        st.pyplot(fig); plt.close()
        st.caption("💡 **B3: 83%** sampel 'BP > NJ'. **Anorganik: 69%** beban tinggi. **Organik: 0%** beban tinggi — semua organik nilai jualnya ≥ biaya proses. Organik adalah kategori paling viable secara ekonomi untuk industri pengolahan sampah.")

    with c2:
        st.subheader("Violin Plot Cost Burden")
        kats_v = [k for k in CATLIST if k in df_f['Kategori'].unique()]
        fig, ax = plt.subplots(figsize=(8,5))
        parts = ax.violinplot([df_f[df_f['Kategori']==k]['cost_burden'].dropna() for k in kats_v],
                               positions=range(1, len(kats_v)+1), showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], [CAT_COLOR[k] for k in kats_v]):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(kats_v)+1)); ax.set_xticklabels(kats_v)
        ax.axhline(0, color='black', linestyle='--', lw=1.5, alpha=0.6, label='Break-even (0)')
        ax.axhline(2, color='red', linestyle=':', lw=1.5, alpha=0.6, label='High-cost threshold (2)')
        ax.set_ylabel('Cost Burden'); ax.set_title('Distribusi Cost Burden (Violin Plot)'); ax.legend(fontsize=9)
        st.pyplot(fig); plt.close()
        st.caption("💡 B3 memiliki distribusi cost_burden paling sempit di atas 2 (high-cost zone) — semua B3 menanggung beban berat dan seragam. Organik memusat di bawah nol. Anorganik paling tersebar — ada yang sangat untung dan sangat rugi.")

    st.subheader("🎯 Kuadran: Nilai Jual Mentah vs Biaya Proses Industri")
    jenis_agg = df_f.groupby(['Nama Sampah','Kategori']).agg(
        nj=('nj_encoded','mean'), bp=('bp_encoded','mean'),
        cb=('cost_burden','mean'), emisi=('Emisi CO2e (kg)','mean')).reset_index()
    jenis_agg['high_cost'] = jenis_agg['cb'] >= 2

    fig, ax = plt.subplots(figsize=(14,7))
    np.random.seed(42)
    for cat, color in CAT_COLOR.items():
        sub = jenis_agg[jenis_agg['Kategori']==cat]
        ax.scatter(sub['nj']+np.random.normal(0,0.05,len(sub)),
                   sub['bp']+np.random.normal(0,0.05,len(sub)),
                   s=sub['emisi']*40+40, alpha=0.75, color=color, edgecolors='white', lw=0.8, label=cat)
    for _, row in jenis_agg[jenis_agg['high_cost']].iterrows():
        ax.annotate(row['Nama Sampah'], xy=(row['nj'], row['bp']),
                    xytext=(row['nj']+0.08, row['bp']+0.05),
                    fontsize=7.5, color='#7f0000', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))
    xd = np.linspace(0.5, 4.5, 100)
    ax.plot(xd, xd+2, 'r--', lw=1.5, alpha=0.7, label='High-cost threshold (BP−NJ≥2)')
    ax.plot(xd, xd, 'k--', lw=1, alpha=0.4, label='Break-even (BP=NJ)')
    ax.fill_between(xd, xd+2, 4.8, alpha=0.06, color='#e74c3c')
    ax.text(1.1, 4.5, 'Zona High-Cost Burden', fontsize=9, color='#c0392b', style='italic')
    ax.set_xticks([1,2,3,4]); ax.set_xticklabels(['Rendah\n<500','Sedang\n500-2k','Sedang/Tinggi\n2k-4k','Tinggi\n>5k'])
    ax.set_yticks([1,2,3,4]); ax.set_yticklabels(['Rendah\n<500','Sedang\n500-3k','Tinggi\n>2k','Sgt Tinggi\n>5k'])
    ax.set_xlabel('Nilai Jual Mentah ke Pabrik (Rp/kg)', fontsize=11)
    ax.set_ylabel('Biaya Proses Industri (Rp/kg)', fontsize=11)
    ax.set_title('Kuadran: Nilai Jual Mentah vs Biaya Proses\n(ukuran titik = emisi CO₂e | label merah = high-cost burden)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left'); ax.set_xlim(0.5, 4.8); ax.set_ylim(0.5, 4.8)
    st.pyplot(fig); plt.close()

    hcb_list = jenis_agg[jenis_agg['high_cost']].sort_values('cb', ascending=False)
    st.subheader(f"⚠️ Jenis Sampah High-Cost Burden ({len(hcb_list)} jenis)")
    st.dataframe(hcb_list[['Nama Sampah','Kategori','nj','bp','cb']].rename(
        columns={'nj':'NJ Encoded','bp':'BP Encoded','cb':'Cost Burden'}),
        use_container_width=True)
    st.caption("💡 **Kaca** mendominasi high-cost: nilai jual mentah sangat rendah (kaca terkontaminasi sulit dijual) + biaya peleburan sangat tinggi (>1400°C). **Plastik sulit daur ulang** (kresek, bungkus snack): pasar hampir tidak ada, butuh mesin ekstrusi khusus. **Elektronik B3 kecil**: mengandung logam berat, butuh fasilitas disposal khusus yang sangat mahal.")

    st.info("📌 **Kesimpulan Q2:** B3 menanggung cost_burden tertinggi secara konsisten. 13 jenis sampah (kaca, plastik sulit, elektronik kecil) masuk zona high-cost burden. Dibutuhkan kebijakan EPR dan subsidi selektif untuk menjaga industri pengolahan sampah tetap berjalan.")

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
elif page == "🔬 Feature Engineering":
    st.title("🔬 Feature Engineering — Fitur Komposit untuk Pemodelan")
    st.markdown("Tiga fitur baru yang lebih informatif untuk model ML:")

    cols_fe = st.columns(3)
    cols_fe[0].metric("⚠️ Avg Env Impact Score", f"{df_f['env_impact_score'].mean():.3f}", help="Dampak lingkungan (emisi+urai+terurai)")
    cols_fe[1].metric("💸 Avg Econ Burden Score", f"{df_f['econ_burden_score'].mean():.3f}", help="Beban biaya ekonomi industri")
    cols_fe[2].metric("♻️ Avg Recyclability Index", f"{df_f['recyclability_index'].mean():.3f}", help="Kemudahan & potensi daur ulang")

    st.subheader("Perbandingan Fitur Baru per Kategori")
    fe_tbl = df_f.groupby('Kategori')[['env_impact_score','econ_burden_score','recyclability_index']].mean().round(3)
    st.dataframe(fe_tbl, use_container_width=True)
    st.caption("💡 B3 mendominasi econ_burden_score (beban ekonomi tertinggi). Organik memiliki recyclability_index tertinggi dan econ_burden_score terendah. Anorganik bervariasi lebar mencerminkan heterogenitas jenisnya.")

    st.subheader("Bubble Chart: Dampak Lingkungan vs Beban Ekonomi")
    fig, ax = plt.subplots(figsize=(14,7))
    fe_agg = df_f.groupby(['Nama Sampah','Kategori']).agg(
        env=('env_impact_score','mean'), econ=('econ_burden_score','mean'),
        ri=('recyclability_index','mean')).reset_index()
    for cat, color in CAT_COLOR.items():
        if cat in df_f['Kategori'].unique():
            sub = fe_agg[fe_agg['Kategori']==cat]
            ax.scatter(sub['econ'], sub['env'], s=sub['ri']*400+50, alpha=0.65, color=color,
                       label=cat, edgecolors='white', lw=0.8)
    ax.axhline(fe_agg['env'].mean(), color='gray', linestyle='--', alpha=0.5, lw=1)
    ax.axvline(fe_agg['econ'].mean(), color='gray', linestyle=':', alpha=0.5, lw=1)
    ax.set_xlabel('Economic Burden Score (0=ringan, 1=berat)', fontsize=12)
    ax.set_ylabel('Environmental Impact Score (0=rendah, 1=tinggi)', fontsize=12)
    ax.set_title('Bubble Chart: Dampak Lingkungan vs Beban Ekonomi\n(Ukuran = Recyclability Index)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    st.pyplot(fig); plt.close()
    st.caption("💡 Sampah 'ideal' ada di kiri-bawah (beban ekonomi rendah, dampak lingkungan rendah) dengan bubble besar (recyclability tinggi). B3 berkelompok di kanan dengan econ_burden tinggi.")

    feat = st.selectbox("Lihat distribusi fitur:", ['env_impact_score','econ_burden_score','recyclability_index'])
    fig, ax = plt.subplots(figsize=(12,4))
    for cat, color in CAT_COLOR.items():
        if cat in df_f['Kategori'].unique():
            ax.hist(df_f[df_f['Kategori']==cat][feat].dropna(), bins=20, alpha=0.65,
                    label=cat, color=color, edgecolor='white')
    ax.set_title(f'Distribusi {feat} per Kategori'); ax.legend()
    st.pyplot(fig); plt.close()

# ── A/B TESTING ───────────────────────────────────────────────────────────────
elif page == "🧪 A/B Testing":
    st.title("🧪 A/B Testing — Efektivitas Metode Pengumpulan")
    st.markdown("""
| Parameter | Detail |
|---|---|
| **H₀** | Tidak ada perbedaan cost_burden antara metode terstruktur dan konvensional |
| **H₁** | Metode terstruktur (Bank Sampah & TPS 3R) mengumpulkan sampah dengan cost_burden **lebih rendah** |
| **Uji** | Mann-Whitney U (non-parametric) |
| **α** | 0.05 |
    """)

    gA = df_f[df_f['Metode Pengumpulan'].isin(['Bank Sampah','TPS 3R'])]['cost_burden'].dropna()
    gB = df_f[df_f['Metode Pengumpulan'].isin(['Pengepul','Drop Box','Pickup Door-to-Door'])]['cost_burden'].dropna()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Group A (n)", f"{len(gA):,}")
    c2.metric("Mean Cost Burden A", f"{gA.mean():.4f}")
    c3.metric("Group B (n)", f"{len(gB):,}")
    c4.metric("Mean Cost Burden B", f"{gB.mean():.4f}")

    if len(gA) > 0 and len(gB) > 0:
        stat, pval = mannwhitneyu(gA, gB, alternative='less')
        cs, cp = st.columns(2)
        cs.metric("U Statistic", f"{stat:.2f}")
        cp.metric("P-value", f"{pval:.6f}")
        if pval < 0.05:
            st.success(f"✅ **Tolak H₀** — p={pval:.4f} < 0.05. Bank Sampah & TPS 3R mengumpulkan sampah dengan cost_burden lebih rendah secara signifikan!")
            st.info("💡 Ini konsisten dengan misi Bank Sampah yang berfokus pada sampah bernilai ekonomi tinggi (kertas, botol, logam) yang memiliki cost_burden rendah. Metode konvensional cenderung tidak selektif sehingga banyak sampah high-cost burden ikut terkumpul.")
        else:
            st.warning(f"❌ **Gagal tolak H₀** — p={pval:.4f} ≥ 0.05")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(gA, bins=10, alpha=0.7, color='#2ecc71', edgecolor='white',
                label=f'Group A (Bank Sampah & TPS 3R)\nMean={gA.mean():.2f}')
        ax.hist(gB, bins=10, alpha=0.7, color='#e74c3c', edgecolor='white',
                label=f'Group B (Konvensional)\nMean={gB.mean():.2f}')
        ax.axvline(gA.mean(), color='#27ae60', linestyle='--', lw=2)
        ax.axvline(gB.mean(), color='#c0392b', linestyle='--', lw=2)
        ax.legend(fontsize=8); ax.set_title('Distribusi Cost Burden A vs B')
        st.pyplot(fig); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(8,4))
        bp3 = ax.boxplot([gA, gB], labels=['Group A\n(Terstruktur)','Group B\n(Konvensional)'], patch_artist=True)
        bp3['boxes'][0].set_facecolor('#2ecc71'); bp3['boxes'][0].set_alpha(0.7)
        if len(bp3['boxes']) > 1: bp3['boxes'][1].set_facecolor('#e74c3c'); bp3['boxes'][1].set_alpha(0.7)
        ax.set_ylabel('Cost Burden'); ax.set_title('Boxplot Perbandingan Cost Burden')
        st.pyplot(fig); plt.close()

# ── DATA DICTIONARY ───────────────────────────────────────────────────────────
elif page == "📋 Data Dictionary":
    st.title("📋 Data Dictionary & Kesiapan Pemodelan")

    try:
        base = os.path.dirname(os.path.abspath(__file__))
        dd = pd.read_csv(os.path.join(base, '../data/data_dictionary.csv'))
        st.dataframe(dd, use_container_width=True, height=500)
    except Exception as e:
        st.warning(f"data_dictionary.csv tidak ditemukan: {e}")

    st.subheader("📦 Kesiapan Dataset untuk ML")
    mc = ['Dapat Terurai','Daur Ulang','Nilai Jual','waktu_urai_bulan','kesulitan_encoded',
          'Emisi CO2e (kg)','nj_encoded','bp_encoded','cost_burden','high_cost_burden',
          'env_impact_score','econ_burden_score','recyclability_index']
    av = [c for c in mc if c in df.columns]
    c1,c2,c3 = st.columns(3)
    c1.metric("Fitur Tersedia", len(av))
    c2.metric("Missing Values", df[av].isnull().sum().sum())
    c3.metric("Duplikat", df[av].duplicated().sum())
    st.success("✅ Dataset siap untuk: Klasifikasi Gambar (CNN), Prediksi high_cost_burden, Prediksi kategori_urai, Klasterisasi.")

st.markdown("---")
st.markdown("<center><small>© 2025 Trashkara Dashboard | Analisis Data Sampah Indonesia</small></center>",
            unsafe_allow_html=True)
