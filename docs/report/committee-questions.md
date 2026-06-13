# Întrebări și observații pentru apărarea tezei

Listă obținută citind toate cele șapte capitole și verificând afirmațiile-cheie
față de cod (`cic-iot2023-detection-system` și `ids-frontend`). Referințele
`fișier:linie` indică sursa de cod, dacă nu e marcat altfel cu „cap. N”.

---

## A. Contradicții lucrare–cod (un comitet care rulează demo-ul le va prinde)

1. **Povestea CICFlowMeter este inversul a ceea ce face codul.** Cap. 5 (FR-D1,
   tech stack) și cap. 6 §Capture-to-CSV Bridge descriu apelarea CICFlowMeter ca
   subproces (`CICFLOWMETER_HOME`, backend-uri Java/Python) tocmai pentru a *evita*
   reimplementarea extracției. Codul nu conține nicio integrare CICFlowMeter;
   `ids/runtime/extractor.py:1-16` este un extractor custom bazat pe DPKT al cărui
   docstring spune explicit „The CIC-IoT-2023 model was NOT trained on CICFlowMeter
   features.” Abordarea din cod este de fapt cea *corectă* (cap. 2 însuși explică
   faptul că feature-urile setului de date vin din DPKT, nu din CICFlowMeter — output-ul
   CICFlowMeter ar produce train/serve skew), deci povestea reală e mai puternică decât
   cea scrisă, dar întreaga secțiune despre bridge, rațiunea FR-D1 și paragraful de
   tech stack descriu o componentă care nu există.

2. **Gradio nu există nicăieri în codebase**, dar apare în FR-D6, cap. 6 §Backend
   („Gradio + FastAPI”, dropdown de model, două taburi de upload), secțiunea de
   deployment („Gradio tab as fallback UI”) și dovada de acceptanță („using both the
   local Gradio interface and the React dashboard”). Backend-ul este doar FastAPI
   (`ids/apps/analyzer/app.py`), iar Dockerfile rulează `python -m ids.apps.analyzer.app`.

3. **Random Forest / XGBoost nu sunt niciodată servite, dar lucrarea pretinde un
   screenshot al unei clasificări RF.** `app.py:26` descoperă doar
   `ids_dnn_*_*class.pth`; modelele tree nu sunt persistate pe disc
   (`ids/training/__init__.py:106-120` le ține în memorie); câmpul `model_type` din
   formular este returnat ca ecou, nu folosit pentru selecție (`app.py:130-156`).
   Totuși legenda Figurii `app_attack` și textul din cap. 6 spun că SYN flood-ul a fost
   „classified under the temporal-split 8-class Random Forest variant” — imposibil cu
   acest backend — iar cele trei carduri de model din frontend (`src/app/data/models.ts`)
   sugerează o alegere fără efect. Propriul TODO semnalează asta; este cea mai
   periculoasă afirmație la apărare („arată-mi rezultatul Random Forest live”).

4. **Cap. 6 §Confidence and Label Decoding spune că scorul de încredere servit „is not
   a calibrated probability”**, în timp ce `ids/runtime/predictor.py:43-72` încarcă
   `temperature_scaling.joblib` și aplică `softmax(logits/T)`, iar cap. 4 + dovada de
   acceptanță descriu scorul servit ca fiind temperature-scaled. Contradicție internă
   directă (deja marcată cu TODO).

5. **Benchmark de latență: textul spune 10.000 de treceri; `ids/core/config.py:80`
   spune `N_RUNS = 1000`** (warmup 100 se potrivește). Apare în cap. 3 §Latency
   Benchmark și FR-P8. (Marcat cu TODO.)

6. **Aritmetica de pruning al feature-urilor se contrazice cu sine însăși și cu codul.**
   Cap. 4 spune că cele două perechi coliniare „were retained” și că 14 feature-uri *flag*
   cad sub pragul de varianță (39 − 14 = 25). Cap. 7 spune „14 near-zero-variance
   indicators and two perfectly collinear continuous pairs” (= 16, ceea ce dă 23, nu 25).
   Codul (`config.py:41-64`) elimină **12** indicatori cu varianță mică plus **2**
   coloane continue redundante (`Tot size`, `Variance`) = 14 în total → 25. Deci
   paragraful „pairs were retained” din cap. 4 e fals față de cod, iar numărătorile din
   cap. 7 sunt greșite. Va fi prins de oricine compară lista de 39 de coloane cu cea de 25.

7. **TODO-urile despre auth/persistență sunt expirate în direcția opusă (veste bună).**
   `@supabase/supabase-js` e în `package.json:42`, `src/lib/supabase.ts`, `useAuth.ts`
   și `useAnalyses.ts` cablează sign-in/up/out și persistența istoricului. FR-D7 și
   cap. 6 §Auth se potrivesc acum cu codul; cele patru comentarii `TODO(verify-vs-code)`
   care spun că auth „not implemented yet” (cap. 5, 6, 7) pot fi rezolvate, nu doar
   atenuate.

8. **Numele fișierelor sample din testul de acceptanță nu se potrivesc cu discul** —
   lucrarea spune `demo_benign.csv` / `demo_syn_flood.csv`; pe disc sunt
   `data/samples/sample_benign_browsing.csv` / `sample_syn_flood.csv`. (Marcat cu TODO.)

9. **Contractul de artefacte nu mai înseamnă „exact trei artefacte”.** Cap. 6
   §Artefact Contract pretinde că partea de serving consumă exact triada
   scaler/encoder/weights, dar calea de serving necesită și `feature_columns.joblib`
   (încărcat de `extractor.py:111` și de sampler) și `temperature_scaling.joblib`
   (`predictor.py:43`). Afirmația „drop three files and restart” e falsă așa cum e scrisă.

---

## B. Un subsistem mare, funcțional, pe care lucrarea nu îl descrie deloc

10. **Live monitor-ul (`ids/apps/monitor/`) lipsește din capitolele 5–6**, deși e
    promis în cap. 1 („scripted attacks are launched against a controlled target and
    the resulting flows are classified in near real time”, contribuția 5). Codul
    implementează: captură live AF_PACKET / replay pcap (`capture.py`), extracție de
    feature-uri pe ferestre tumbling (`windower.py`), un design cu două modele (gate
    binar de 2 clase + clasificator de familie pe 8 clase, `detector.py:38-39`), o
    politică de ban (prag de încredere + N ferestre malițioase consecutive + allowlist,
    `enforcement.py:32-62`), **enforcement real prin nftables** care aruncă traficul la
    nivel de kernel (`enforcement.py:126-165`) și un flux de evenimente SSE consumat de
    `LiveMonitorPage` din frontend. Cap. 2 §IDS-vs-IPS chiar descrie exact acest punct
    de mijloc („detect beside the path, respond with a firewall rule”) ca teorie — fără
    să spună că sistemul îl implementează. Este partea cea mai impresionantă a
    construcției și e invizibilă în capitolele de cerințe, design, tabelul de use-case
    și testare.

11. **Tooling-ul de atac și site-ul victimă sunt nedocumentate**: `attacks/recon.sh`
    (nmap SYN scan), `synflood.sh`/`udpflood.sh` (hping3), `spoofed_flood.sh`
    (`--rand-source`, care demonstrează intenționat limitarea ban-ului pe IP sursă — o
    ilustrare live perfectă a argumentului despre atacuri volumetrice din cap. 2),
    `swarm_entrypoint.sh` (swarm multi-container cu IP-uri sursă distincte — adică o
    demonstrație reală DDoS-vs-DoS) și `mock_site/` (site-ul protejat care se abonează la
    evenimentele detectorului). Un comitet va întreba „cum ai generat atacurile pentru
    demo?”, iar răspunsul trăiește acum doar în repo.

12. **Explicabilitatea SHAP este implementată și servită, dar nemenționată.**
    `ids/runtime/explain.py` (GradientExplainer, contribuții semnate top-k), cablat în
    `/api/classify` pentru varianta temporal/8-class (`app.py:47-63, 88-109, 152-153`),
    randat ca panou de top-features pe `ResultsPage.tsx:51-55`; live monitor-ul folosește
    intenționat un proxy ieftin de saliency în loc, din motive de latență. Cap. 4 citează
    literatura XAI-for-IDS pentru a justifica permutation importance, dar explicația
    per-predicție din producție — o contribuție mult mai puternică — primește zero cuvinte.
    Trade-off-ul de latență (SHAP pe calea de upload, proxy pe calea live) este exact
    genul de decizie de design pentru care există capitolul de aplicație.

13. **Nepotrivire de scope la frontend**: lucrarea descrie un dashboard + ecran de
    analiză; aplicația are paginile Landing, Login/Register, Dashboard, Analysis, Results
    (cu SHAP), **LiveMonitor** și **Comparison**. Tabelul de use-case (UC-1…UC-4) nu are
    niciun use-case de monitorizare live.

14. **Modul simulate al demo-ului exclude discret clasele slabe.**
    `ids/data/sampler.py:19-23` reia doar familiile „green” (Benign, DDoS, DoS, Mirai,
    Recon) „the model detects reliably — Web/Spoofing excluded — see thesis methodology”.
    Având în vedere NFR-8 („honest reporting”), această curatare ar trebui dezvăluită în
    lucrare, altfel un membru al comitetului care citește codul va întreba de ce demo-ul
    evită clasele la care modelul e slab.

---

## C. Întrebări metodologice probabile din partea comitetului (verificabile în cod)

15. **Este split-ul „temporal” chiar cronologic? Sortarea e lexicografică.**
    `preprocessing.py:40` folosește `sorted(set(...))` pe nume de fișiere precum
    `DDoS-ICMP_Flood.pcap.csv, ...Flood1, ...Flood10, ...Flood11, …, ...Flood19,
    ...Flood2, ...Flood20` (confirmat pe disc). Ordinea lexicografică pune `Flood2`
    *după* `Flood19`, iar fișierul fără sufix primul. Afirmația din cap. 1 că „the test
    data is strictly newer than the training data” nu e susținută de această sortare; ai
    nevoie de o cheie de sortare numerică naturală (și de dovada că ordinea sufixelor =
    ordinea capturii) sau de o afirmație slăbită. Este contribuția metodologică principală,
    deci așteaptă-te la întrebarea asta.

16. **Pentru clasele cu o singură captură, fallback-ul „temporal” e de fapt aleator.**
    Folderele cu <3 CSV-uri cad pe un split pe ordinea rândurilor (`preprocessing.py:57-65`),
    dar ordinea rândurilor nu e păstrată în amonte: `ingest.py:47` deduplichează cu
    `unique()` (fără `maintain_order=True`, deci polars nu garantează ordinea) și
    `ingest.py:51` subeșantionează cu `shuffle=True`. Pe copia de pe disc, **Benign are
    exact un CSV** (`Benign_Final/BenignTraffic3.pcap.csv`) — deci partea benignă a
    rezultatului *binar* temporal de bază e împărțită pe rânduri amestecate, adică aleator.
    Același lucru pentru Recon-PortScan și clasele web rare. „Cât de temporal e de fapt
    0.90-ul binar?” e o întrebare corectă și dăunătoare dacă nu e abordată.

17. **Ce date benigne au fost de fapt ingerate?** Cap. 2 pretinde 1,1M de eșantioane
    benigne și 169 de CSV-uri; pe disc sunt 217 CSV-uri în 34 de foldere de clase, cu
    benign redus la un singur fișier de captură. Dacă statisticile de ingestie din Figura
    `ingest_waterfall` au fost produse dintr-o copie diferită/mai completă a datelor decât
    cea de pe disc acum, reproductibilitatea (NFR-3, FR-P1) e subminată; oricum, lucrarea
    ar trebui să spună exact ce release/conversie a fost folosit (CSV-uri per-pcap, nu cele
    169 de părți oficiale unite).

18. **Comparația promisă random-vs-temporal nu e livrată niciodată.** Cap. 1 și cap. 3
    construiesc argumentul de leakage („temporal F1 expected a few points below random F1;
    that gap is the honest generalisation story”; split-ul random „reported only for
    parity”), iar FR-P3 cere toate trei split-urile — dar cap. 4 raportează *doar*
    split-ul temporal, `config.py:68` are `SPLITS_TO_RUN = ['temporal']`, și doar
    artefactele temporale există în `models/`. Singura afirmație cantitativă care ar dovedi
    teza de leakage (gap-ul random–temporal) lipsește. Cap. 7 concede doar split-ul
    per-capture ca lucru viitor; și split-ul random e absent în tăcere.

19. **Rezultatele de latență sunt promise dar niciodată raportate.** Cap. 3 §Latency
    Benchmark și FR-P8/NFR-1 definesc p50/p95/p99 per batch size ca livrabil; cap. 4 nu
    conține nicio secțiune de latență. Singurele numere sunt anecdotice („8 ms for 50
    flows”, „under 10 ms” în cap. 7). `ids/training/benchmark.py` produce exact tabelul de
    care ai nevoie — rulează-l și adaugă-l.

20. **Rezultatele binare sunt subțiri raportat la statutul lor de „headline”.** NFR-2 e
    îndeplinit pe taskul binar, dar cap. 4 dă numerele binare doar în treacăt (≈0.90/≈0.92
    în paragraful de comparație a modelelor) fără un tabel binar per-model echivalent cu
    Tabelul `model_comparison_table`, iar matricea de confuzie binară e doar pentru MLP.
    Așteaptă-te la „unde e tabelul binar?”

21. **HPO a fost mai îngust decât e descris.** `tune.py` rulează 15 trials doar pe taskul
    **binar** (`tune_mode='2'`), cu un subeșantion **uniform** `rng.choice` (`tune.py:42`)
    — lucrarea spune „stratified” — iar configurația câștigătoare e refolosită neschimbată
    pentru modelul de 8 clase. De asemenea `tune.py:79` construiește modelul cu `N_FEATURES`
    (39), implicând că search-ul a rulat pe setul de feature-uri pre-pruning, în timp ce
    toate modelele raportate folosesc 25 de feature-uri. „Ai tunat pe alt task și pe altă
    dimensionalitate de input decât modelul pe care îl raportezi” e un push corect al
    comitetului; fraza din lucrare „this configuration is then retrained…” acoperă asta.

22. **Pruning-ul bazat pe varianță poate fi șters semnalul claselor minoritare.**
    `DROPPED_LOW_VAR` (`config.py:45-58`) include `ARP` și `ICMP` — dar Spoofing *este*
    spoofing ARP/DNS (cea mai slabă clasă a ta, recall 0.46, 46% prezis Benign), iar cea
    mai mare clasă unică e DDoS-ICMP_Flood. Varianța aproape-zero pe un eșantion dominat de
    DDoS/Mirai e de așteptat pentru un flag informativ doar pentru o clasă <1%; pragul de
    0.01 elimină deci sistematic exact indicatorii claselor rare. Fie justifică (de ex.
    coloanele ARP/ICMP din CSV-ul public sunt genuin degenerate — merită verificat), fie
    recunoaște-l ca posibilă cauză a eșecului pe Spoofing. Întrebare puternică și foarte
    probabilă.

23. **Coliziunile de duplicate între clase sunt nemăsurate.** Deduplicarea e per-folder
    pe feature-uri (`ingest.py:47`), deci vectori de feature-uri identici la byte care apar
    sub etichete *diferite* (foarte plauzibil pentru SYN flood DoS vs DDoS) supraviețuiesc
    în ambele clase. Asta pune un plafon dur pe separarea DoS/DDoS și ar putea explica o
    parte din confuzia mutuală de 26–31%; cuantificarea ei ar întări argumentul „the data,
    not the model” pe care discuția deja îl face.

24. **Numerele de confuzie DoS/DDoS diferă între capitole**: tabelul din cap. 4 spune
    31% / 26%; cap. 7 spune „up to 36% cross-confusion”.

25. **Cum a fost validat extractorul custom față de distribuția de antrenare?** Cap. 5
    argumentează că reimplementarea „would risk numerical drift” — codul apoi reimplementează
    extracția, cu presupuneri documentate explicit acolo unde articolul CIC tace
    (`extractor.py:9-16`: ferestre tumbling vs sliding, perechi de hosturi neordonate,
    lungime de frame, agregare pe fracție a flag-urilor, ferestre parțiale finale).
    `ids/runtime/validate_extractor.py` există; rezultatele lui (comparația de distribuție a
    feature-urilor față de CSV-urile oficiale) aparțin tezei, altfel testul de acceptanță e
    singura dovadă că feature-urile live sunt in-distribution.

26. **Dimensiunea ferestrei la inferență e fixată la 10, dar datele de antrenare au amestecat
    10 și 100.** Setul de date a folosit ferestre de 100 de pachete pentru clasele de
    flooding (`extractor.py:4-5`), ceea ce e informație per-clasă indisponibilă la inferență;
    `extract_features` are default `window=10`. Cum `Number` (numărul de pachete) e cel mai
    important feature al tău (cap. 4), iar `Rate`, count-urile și `Tot sum` scalează toate cu
    lungimea ferestrei, „modelul învață parțial parametrizarea extracției în loc de trafic, și
    ce face servirea la window=10 fluxurilor de flood extrase la 100 în antrenare?” e o
    întrebare ascuțită.

27. **Parametrii politicii de ban sunt neexplicați nicăieri.** Monitorul banează pe încredere
    calibrată ≥ prag pe N ferestre consecutive (`enforcement.py:32-62`). Dacă sistemul live
    intră în teză (punctul 10), alegerea pragului/N — și relația lui cu analiza de calibrare
    și trade-off-ul matricei de confuzie binare — are nevoie de un paragraf; aici e exact
    locul unde munca de temperature-scaling dă roade, iar legătura nu e trasă niciodată acum.

28. **Detalii mai mici verificabile**: numărul de bin-uri ECE (lucrarea: 15 bin-uri de
    lățime egală) ar trebui confirmat față de `calibration.py`; afirmația „heaviest class
    weight >50× the lightest” (cap. 3) e verificabilă din artefactele run-ului; tracking-ul
    W&B (`tune.py:106`, `tracking.py`) e o dependență ascunsă în povestea „reproducible,
    one-command”; iar promisiunea FR-D5 „never persisted” are nevoie de o frază despre live
    monitor, care observă continuu trafic în loc să proceseze upload-uri individuale.

---

## Observație generală

Capitolele scrise 5–6 descriu o încarnare anterioară a sistemului (Gradio + CICFlowMeter +
doar upload), în timp ce codebase-ul real a evoluat în ceva mai substanțial (extractor custom
fidel, pipeline live captură-la-ban, SHAP, auth funcțional). Cele mai multe goluri de mai sus
se rezolvă rescriind capitolele de aplicație în jurul sistemului real — care e și sistemul mai
defensabil — și re-rulând/raportând trei lucruri: split-ul temporal cu sortare numerică
naturală, comparația cu split-ul random și tabelul de latență.
