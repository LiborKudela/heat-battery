# Parametry `mesh_p` a `sim_p` (Example_05)

Detailní popis všech parametrů, které se v příkladu **Example_05** předávají do
slovníků `mesh_p` (geometrie a síťování v Gmsh) a `sim_p` (simulace
`C3_passive.solve_unsteady`).

Hodnoty parametrů jsou v [`run_example.py`](run_example.py) zabaleny do
`ParameterGrid` / `ParameterList` (parametrická studie) a v
[`run_example_size5.py`](run_example_size5.py) jsou zadány jako jeden
konkrétní případ. Zde popisuji **co každý parametr znamená**, ne jeho
konkrétní hodnotu.

> Skutečným zdrojem pravdy zůstává kód:
> [`geometry_v7.py`](geometry_v7.py) (geometrie),
> [`model.py`](model.py) (simulace, ko-simulace) a
> [`heat_battery/simulations/simulation_base.py`](../../heat_battery/simulations/simulation_base.py)
> (`Simulation.solve_unsteady`).

---

## Obsah

1. [Schéma geometrie](#1-schéma-geometrie)
2. [Parametry `mesh_p`](#2-parametry-mesh_p)
   1. [Obecné parametry generování sítě](#21-obecné-parametry-generování-sítě)
   2. [Geometrie zásobníku a izolace](#22-geometrie-zásobníku-a-izolace)
   3. [Topné patrony (cartridge)](#23-topné-patrony-cartridge)
   4. [Žebrový rozváděč patrony (cartridge spreader)](#24-žebrový-rozváděč-patrony-cartridge-spreader)
   5. [Trubky výměníku (THT / THP)](#25-trubky-výměníku-tht--thp)
   6. [Žebrový rozváděč trubek (THT spreader)](#26-žebrový-rozváděč-trubek-tht-spreader)
   7. [Segmentace povrchů THP](#27-segmentace-povrchů-thp)
   8. [Materiály](#28-materiály)
3. [Parametry `sim_p`](#3-parametry-sim_p)
   1. [Obecné parametry řešiče (`Simulation.solve_unsteady`)](#31-obecné-parametry-řešiče-simulationsolve_unsteady)
   2. [Tolerance, počáteční podmínky a časové řízení](#32-tolerance-počáteční-podmínky-a-časové-řízení)
   3. [Výstupy: XDMF, sondy, checkpointy](#33-výstupy-xdmf-sondy-checkpointy)
   4. [Parametry specifické pro `C3_passive` (ko-simulace s budovou)](#34-parametry-specifické-pro-c3_passive-ko-simulace-s-budovou)

---

## 1. Schéma geometrie

Geometrie je **válcový sektor** o úhlu `2π / cartridge_n` (využívá rotační
periodicity). Skládá se z:

- **vnitřního válce** (zásyp / sand) o průměru a výšce `size`,
- **vnější vrstvy izolace** o tloušťce `t_insulation` (radiální i axiální),
- **topných patron** (`cartridge_*`) zapuštěných v zásypu,
- **trubek výměníku** (`tht_*` / `thp_*`) procházejících zásypem (nebo izolací,
  je-li `tht_in_sand=False`),
- **žebrových rozváděčů** kolem patron a kolem trubek (zlepšují přestup tepla).

Výstupní okrajové plochy:

| Název v Gmsh                | Význam                                                           |
| --------------------------- | ---------------------------------------------------------------- |
| `Outer_surface`             | Vnější plášť (izolace) — pasivní ztráty tepla do místnosti.      |
| `THP_surface_<i>_<j>`       | Vnitřní povrch *i*-té trubky, *j*-tého axiálního segmentu.       |

Materiálové podoblasti: `Sand`, `Insulation`, `Cartridge`, `Cartridge_spreader`,
`THT_spreader`.

---

## 2. Parametry `mesh_p`

Všechny parametry zde popsané se předávají funkci `build_geometry` v
[`geometry_v7.py`](geometry_v7.py).

### 2.1 Obecné parametry generování sítě

| Parametr                    | Typ      | Význam                                                                                                                                |
| --------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                      | `str`    | Název modelu (a souborů `<name>.msh`, `<name>.step`, `<name>.ad`).                                                                    |
| `dir`                       | `str`    | Adresář, kam se síť uloží / odkud se načte (např. `meshes/C3_passive`).                                                               |
| `verbosity`                 | `int`    | Úroveň výpisu Gmsh (0 = ticho, 1 = informativní, 2 = debug). V `run_example.py` zabaleno do `NoNumericalEffect` — neovlivní hash úlohy. |
| `fltk`                      | `bool`/`str`/`list` | Otevře Gmsh GUI v zadané fázi (`'premesh'`, `'postmesh'`). `False` = bez GUI. Také `NoNumericalEffect`.                  |
| `mesh_size_max`             | `float`  | Globální horní mez velikosti elementu sítě (m). Default `0.1`.                                                                        |
| `mesh_size_from_curvature`  | `int`    | Počet elementů, kterými má Gmsh aproximovat oblouk 2π (vyšší = jemnější síť na zakřivených plochách). Default `16`–`18`.              |

> Pozn.: `symmetry` (v signatuře `build_geometry`) v Example_05 nepoužívá —
> geometrie je vždy „plná" výseč o úhlu `2π / cartridge_n`.

### 2.2 Geometrie zásobníku a izolace

| Parametr        | Typ     | Význam                                                                                                  |
| --------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `size`          | `float` | Průměr **i** výška vnitřního válce zásypu (m). V parametrické studii: `[1, 3, 5]` m.                    |
| `t_insulation`  | `float` | Tloušťka izolační vrstvy (m), aplikovaná radiálně i axiálně okolo zásypu. Studie: `[0.5, 1.0]` m.       |

Vnější válec má tedy poloměr `size/2 + t_insulation` a výšku
`size + 2·t_insulation`.

### 2.3 Topné patrony (cartridge)

Patrony jsou svisle osazené válečky elektrického topení, rozmístěné po
kruhové dráze v zásypu.

| Parametr               | Typ     | Význam                                                                                                                                              |
| ---------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cartridge_n`          | `int`   | Počet patron na celou geometrii (zároveň i počet úhlových výsečí). Studie: `[4, 10]`.                                                               |
| `cartridge_d_ratio`    | `float` ∈⟨0,1⟩ | Radiální poloha patrony jako podíl prostoru mezi středem zásypu a místem trubek. `0` = u středu, `1` = u trubek.                            |
| `cartridge_diameter`   | `float` | Průměr (m) topné patrony bez žeber. Default `0.014` m.                                                                                              |
| `cartridge_h_ratio`    | `float` ∈⟨0,1⟩ | Poměr výšky patrony k výšce zásypu (`size`). `1` = patrona přes celou výšku zásypu.                                                          |

### 2.4 Žebrový rozváděč patrony (cartridge spreader)

Plechová žebra ve tvaru hvězdy zvyšují účinný přestupný povrch z patrony do
zásypu.

| Parametr                                | Typ     | Význam                                                                                                                                                                  |
| --------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cartridge_spreader_lb`                 | `float` | Délka žebra (m), tj. radiální rozměr od povrchu patrony k vnějšímu okraji žebra. Studie: `[0.02, 0.06]` m.                                                              |
| `cartridge_spreader_nb`                 | `int`   | Počet žeber okolo patrony (rozmístěna úhlově rovnoměrně). Default `3`.                                                                                                  |
| `cartridge_spreader_tb`                 | `float` | Tloušťka žebra (m). Default `0.005` m.                                                                                                                                  |
| `cartridge_spreader_mesh_size_min`      | `float` | Minimální velikost elementu sítě **u povrchu** žebra patrony (m). Z toho roste velikost s vzdáleností.                                                                  |
| `cartridge_spreader_mesh_grow_factor`   | `float` | Faktor růstu velikosti elementu se vzdáleností od povrchu žebra patrony (větší = rychleji hrubne).                                                                      |

Velikost elementu je v Gmsh sestavena přes `MathEval` jako
`min(min + grow·F1, …)`, kde `F1` je vzdálenost od povrchu žebra patrony.

### 2.5 Trubky výměníku (THT / THP)

„THT pipes" / „THP" jsou průchozí trubky (Through-Hole Pipes), kterými proudí
vzduch z budovy a odebírá teplo ze zásypu.

| Parametr        | Typ     | Význam                                                                                                                                                                                                       |
| --------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tht_in_sand`   | `bool`  | `True` = trubky jsou uvnitř zásypu (sand). `False` = trubky procházejí izolací (méně obvyklé).                                                                                                              |
| `tht_d`         | `float` | Vnitřní průměr trubky (m). Studie: `[0.04, 0.08]` m. Musí platit `tht_d < t_insulation`.                                                                                                                     |
| `tht_d_ratio`   | `float` ∈⟨0,1⟩ | Relativní radiální poloha trubek ve volném prostoru mezi patronami a vnější hranou zásypu (`tht_in_sand=True`) nebo v izolaci (`tht_in_sand=False`).                                                  |
| `tht_n_ratio`   | `float` ∈⟨0,1⟩ | Jaký podíl dostupného úhlového prostoru má být zaplněn trubkami. Skutečný počet trubek `n_m` se z toho v kódu spočte (alespoň 1).                                                                     |

### 2.6 Žebrový rozváděč trubek (THT spreader)

Žebra okolo trubek zvyšují přestup tepla mezi zásypem a vzduchem v trubce.

| Parametr                            | Typ     | Význam                                                                                                                                          |
| ----------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `tht_spreader_h_ratio`              | `float` ∈⟨0,1⟩ | Poměr výšky žebrové sekce trubky k výšce zásypu (resp. vnějšího válce). `1` = žebra po celé výšce.                                       |
| `tht_spreader_lb`                   | `float` | Délka žebra trubky (m). Studie: `[0.02, 0.06]` m.                                                                                              |
| `tht_spreader_nb`                   | `int`   | Počet žeber okolo trubky. Default `3`.                                                                                                          |
| `tht_spreader_tb`                   | `float` | Tloušťka žebra trubky (m). Default `0.005` m.                                                                                                   |
| `thp_spreader_mesh_size_min`        | `float` | Minimální velikost elementu sítě **u povrchu** žebra trubky.                                                                                    |
| `thp_spreader_mesh_grow_factor`     | `float` | Faktor růstu velikosti elementu se vzdáleností od povrchu žebra trubky.                                                                         |
| `thp_mesh_size_min`                 | `float` | Minimální velikost elementu sítě **u povrchu vnitřního kanálu trubky** (`THP_surface_*`).                                                        |
| `thp_mesh_grow_factor`              | `float` | Faktor růstu velikosti elementu se vzdáleností od povrchu kanálu trubky.                                                                        |

> `tht_*` versus `thp_*`: `tht_*` parametry řídí **geometrii** (rozměry, počty),
> `thp_*` parametry řídí **síť** kolem vnitřních povrchů kanálu (BC plochy).
> V kódu jsou totéž zařízení — pouze konzistentně oddělené konvence.

### 2.7 Segmentace povrchů THP

| Parametr                | Typ   | Význam                                                                                                                                                                       |
| ----------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `thp_surface_segments`  | `int` | Po kolika axiálních segmentech rozdělit vnitřní povrch každé trubky. Každý segment dostane vlastní okrajovou podmínku `THP_surface_<i>_<j>`, takže lze sledovat průběh teploty po výšce trubky. Default `1`, ve studii `10`. |

### 2.8 Materiály

Předávají se jako **jména** materiálů; kód si je vyhledá přes
`materials.get_material_by_name(...)`. Definice materiálů jsou v
`heat_battery/materials/`.

| Parametr                       | Příklad hodnoty           | Význam                                              |
| ------------------------------ | ------------------------- | --------------------------------------------------- |
| `sand_material`                | `"SandTheory"`            | Materiál zásypu (hlavní válec).                     |
| `insulation_material`          | `"Standard_insulation"`   | Materiál izolace (vnější válec).                    |
| `cartridge_material`           | `"Steel04"`               | Materiál topné patrony.                             |
| `cartridge_spreader_material`  | `"Steel04"`               | Materiál žeber kolem patrony.                       |
| `thp_spreader_material`        | `"Steel04"`               | Materiál žeber kolem trubky výměníku.               |

---

## 3. Parametry `sim_p`

Tyto parametry se předávají metodě `C3_passive.solve_unsteady` (definice v
[`model.py`](model.py)). Většina z nich je dále propuštěna do
`Simulation.solve_unsteady` v
[`simulation_base.py`](../../heat_battery/simulations/simulation_base.py).

Časové parametry jsou v sekundách **simulovaného modelu** (ne reálného běhu).

### 3.1 Obecné parametry řešiče (`Simulation.solve_unsteady`)

| Parametr                | Typ      | Význam                                                                                                                              |
| ----------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `verbose`               | `bool`   | `True` = každý krok vypsat tabulku sond. `False` = jen progress bar. V `run_example.py` zabaleno do `NoNumericalEffect`.            |
| `t_max`                 | `float`  | Konec simulace (s). V příkladu `2*365*24*3600` = 2 roky.                                                                            |
| `dt_start`              | `float`  | Počáteční velikost časového kroku (s).                                                                                              |
| `dt_min`                | `float`  | Minimální dovolená velikost kroku (s). Při poklesu pod tuto hranici se simulace přeruší.                                            |
| `dt_max`                | `float`  | Maximální dovolená velikost kroku (s). Default ve studii `1200` s = 20 min.                                                         |
| `force_explicit_terms`  | `bool`   | `True` = vypne implicitní vnější iterace pro Term-y (rychlejší, ale méně přesné při silné ko-simulaci).                             |
| `dt_ctrl_interval`      | `tuple(float, float)` | Dolní a horní hranice pro řízení časového kroku podle `max\|ΔT\|` mezi kroky. Krok se zvětší, když je změna malá; zmenší, když moc velká. |

### 3.2 Tolerance, počáteční podmínky a časové řízení

| Parametr     | Typ      | Význam                                                                                                                |
| ------------ | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `T0`         | `float`  | Uniformní počáteční teplota celé domény (°C). V příkladu `18`.                                                        |
| `h0_T_ref`   | `float`  | Referenční teplota pro výpočet entalpie materiálů (°C). Obvykle stejná jako `T0`.                                     |
| `atol`       | `float`  | Absolutní tolerance Newtonova řešiče SNES. V příkladu `1e-6`.                                                         |
| `rtol`       | `float`  | Relativní tolerance Newtonova řešiče SNES. V příkladu `1e-7`.                                                         |
| `datetime_start` | `str` | „Kotva" pro reálné časové značky sond (formát `'YYYY-MM-DD HH:MM:SS.s'`, **UTC**). Posunuje start meteorologické řady.|

### 3.3 Výstupy: XDMF, sondy, checkpointy

| Parametr                 | Typ      | Význam                                                                                                                                                                          |
| ------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dt_xdmf`                | `float`  | Jak často (s) zapsat plné teplotní pole do XDMF. V `run_example.py` `3600` (každou hodinu), v `run_example_size5.py` `24*3600` (každý den).                                     |
| `xdmf_file`              | `str`/`None` | Název XDMF souboru (např. `'unsteady.xdmf'`). `None` = XDMF nezapisovat. Vyžaduje, aby `result_dir` byl nastaven.                                                          |
| `result_dir`             | `str`    | Cesta k adresáři pro XDMF a CSV sondy (jen v přímém běhu `run_example_size5.py`; v projektu se nastavuje jinak).                                                                |
| `probe_destinations`     | `list[dict]` | Seznam cílů, kam streamovat skalární sondy. Typy `'memory'`, `'csv'`, `'database'`. V `run_example_size5.py` se zapisuje do `unsteady.csv`.                                |
| `checkpoint_dt`          | `float`/`None` | Jak často (s) ukládat plný checkpoint stavu (`adios4dolfinx`). V příkladu `7*24*3600` = jednou za týden simulovaného času. V `run_example.py` zabaleno do `NoNumericalEffect`.|
| `checkpoint_dir`         | `str`/`None` | Adresář pro checkpointy (jen v přímém běhu). Pokud existuje předchozí, simulace **automaticky obnoví stav**.                                                              |
| `load_initial_checkpoint`| `str`/`None` | Cesta k existujícímu checkpoint adresáři, ze kterého se má start obnovit. V `run_example_size5.py` se nastavuje automaticky podle existence `metadata.json`.              |

> **Pozor:** Pokud změníte parametry geometrie nebo simulace, je nutné
> checkpoint adresář **smazat**, jinak se znovu obnoví starý nekonzistentní
> stav.

### 3.4 Parametry specifické pro `C3_passive` (ko-simulace s budovou)

Tyto parametry řídí ko-simulační model **HallC3** (lumped model haly C3 na FSI
VUT Brno) a propojení s elektrickou patronovou sekcí. Definice viz
`HallC3` a `C3_passive.solve_unsteady` v [`model.py`](model.py).

#### Termické řízení haly

| Parametr                     | Typ                  | Význam                                                                                                                                                                                                                                                 |
| ---------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `T_room_ctrl_interval`       | `tuple(float, float)` | Pásmo přípustné změny teploty místnosti `\|ΔT_room\|` mezi kroky (K). Pokud je změna nad horní mez, krok se zmenší; pod dolní mez se zvětší. V `HallC3.adaptation`. V příkladu `(0.1, 0.2)`.                                                          |
| `converge_tol_T_room`        | `float`              | Tolerance pro reziduum implicitního dorovnání teploty místnosti (K) ve vnějších iteracích. V příkladu `0.1` K.                                                                                                                                          |
| `converge_tol_Q_amb`         | `float`              | Tolerance pro reziduum implicitního dorovnání ztráty `Q_amb` (W). V příkladu `10` W.                                                                                                                                                                    |

#### Přestup tepla

| Parametr        | Typ                  | Význam                                                                                                                                                                                                                                                                                  |
| --------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alpha_s`       | `float`              | Součinitel přestupu tepla mezi vnější plochou izolace a vzduchem v místnosti (W/m²·K). Konstantní, v příkladu `5.0`.                                                                                                                                                                  |
| `alpha_m_lims`  | `tuple(float, float)` | Spodní a horní mez součinitele přestupu mezi povrchem trubky výměníku (THP) a proudícím vzduchem (W/m²·K). Skutečná hodnota se v každém kroku **dynamicky dopočítá** tak, aby teplo z THP právě pokrylo deficit topení (equithermal control). V příkladu `(0.1, 20.0)`.            |

#### Meteo a fotovoltaika

| Parametr           | Typ      | Význam                                                                                                                                                                                                                          |
| ------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `location`         | `tuple`  | `(lat, lon, alt)` souřadnice lokality pro stažení meteorologických dat z PVGIS. V příkladu `meteodata.locations['Brno-FME']`. Vlastní lokalitu lze přidat stejným formátem.                                                     |
| `pv_peak`          | `float`  | Špičkový výkon FVE (W). Z poměrného PV signálu se násobí jako `(pv_peak/1000)·P(t)`. V příkladu `30000` (30 kW).                                                                                                                |
| `datetime_start`   | `str`    | Datum začátku simulace pro posun meteorologické řady (`'YYYY-MM-DD HH:MM:SS.s'`, UTC). V příkladu `'2007-6-1 00:00:00.0'`.                                                                                                       |

#### Patrony (PV → cartridge)

| Parametr     | Typ     | Význam                                                                                                                                                                                       |
| ------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Tc_limit`   | `float` | Maximální dovolená teplota patrony (°C). Když ji teplota patrony překročí, kód postupně utlumí přívod elektrického výkonu (`toggle` v `TemperatureLimitedUniformHeatSource`). V příkladu `500`. |

#### Dotápění a max. odběr

| Parametr               | Typ     | Význam                                                                                                                                                                                          |
| ---------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_bivalent_power`   | `float` | Maximální výkon záložního (bivalentního) zdroje v hale (W). Aktivuje se, když úložiště **samo** nedokáže pokrýt aktuální tepelnou ztrátu. V příkladu `30000` (30 kW).                            |
| `max_mem_power`        | `float` | Maximální výkon, který může výměník (THP) odevzdat do vzduchu (W). Pojistka proti nereálným hodnotám `α_m`. V příkladu `30000` (30 kW).                                                          |

---

## Stručná „cheat sheet" parametrické studie

V `run_example.py` se mění tyto osy (kombinace dávají hash → samostatnou
úlohu v PostgreSQL frontě):

| Osa                       | Hodnoty                                            |
| ------------------------- | -------------------------------------------------- |
| `size`                    | `[1, 3, 5]` m                                      |
| `t_insulation`            | `[0.5, 1.0]` m                                     |
| `cartridge_n`             | `[4, 10]`                                          |
| `cartridge_spreader_lb`   | `[0.02, 0.06]` m                                   |
| `tht_d`                   | `[0.04, 0.08]` m                                   |
| `tht_d_ratio`             | `[0.1, 0.15]`                                      |
| `tht_n_ratio`             | `[0.2, 0.4, 0.6]`                                  |
| `tht_spreader_lb`         | `[0.02, 0.06]` m                                   |

Všechny ostatní parametry jsou pevné. `verbosity`, `fltk` a `checkpoint_dt`
jsou ovinuty `NoNumericalEffect(...)` — jejich změna **nezneplatní** již
spočtené úlohy (nejsou součástí podpisu úlohy).

---

*Tento dokument popisuje stav repozitáře v době psaní; při rozporu má
přednost zdrojový kód* (`geometry_v7.py`, `model.py`,
`heat_battery/simulations/simulation_base.py`).
