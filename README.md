"# door-assembly" 
1. Objective

The purpose of this simulation is to model the door assembly line as described in the DALB (Door Assembly Line Balancing) mathematical formulation. The simulation in FlexSim allows us to:

Validate the mathematical model assumptions.

Evaluate performance (throughput, shortages, rework, utilization).

Experiment with demand, rework rates, availability, and overtime scenarios.

2. Model Inputs and Mapping

The mathematical model parameters were mapped into FlexSim objects following the scheme from Data_Table[1].pdf
:

Model Input (Symbol)	Description	FlexSim Location / Implementation

𝑑
𝑡
,
𝐴
𝑡
,
𝜏
𝑡
d
t
	​

,A
t
	​

,τ
t
	​

	Demand, period length, takt time	Source (Interarrival time or Schedule Table)

𝑞
𝑡
,
𝜔
,
𝑣
q
t,ω,v
	​

	Variant mix (L/R, Base/Premium)	Source → OnCreation label “variant”

𝑝
𝑖
,
𝑣
p
i,v
	​

	Base processing time per task & variant	Processor → Process Time

𝑟
𝑖
,
𝛾
𝑖
r
i
	​

,γ
i
	​

	Rework probability and factor	Included in Process Time expression

𝑎
𝑠
,
𝑡
,
𝜔
a
s,t,ω
	​

	Availability of stations	Processor → Downtime/Shift Calendar

𝛿
𝑠
,
𝑡
,
𝜔
δ
s,t,ω
	​

	Setup/changeover losses	Processor → Setup/Changeover Matrix

𝑃
P	Task precedence	Sequential Processor layout

𝐸
𝑖
,
𝑔
𝑖
,
𝑁
𝑚
𝑎
𝑥
E
i
	​

,g
i
	​

,N
max
	​

	Ergonomics, footprint, colocation caps	Static reference (checked externally)
3. Model Structure

Flow Layout:

Items (doors) enter from a Source, flow sequentially through 13 Processors (each representing one assembly task), and exit via a Sink.

Task names: Window Regulator, Latch & Lock, Handle, etc., up to End-of-Line Test.

Variants:

Each arriving door is labeled with one of four variants (L-Base, L-Premium, R-Base, R-Premium).


Processing Times:

Each Processor calculates time based on door variant.

Example (Window Regulator):

Base time from table

Premium time from table

Rework Logic:

Incorporated into Process Time.

Example: with 
𝑟
𝑖
=
values of ri from the table
r
i
	​


𝛾
𝑖
=
values from the table 
γ
i
	​

, the process time becomes:

𝑝
=
𝑝
𝑖
,
𝑣
(normal)
or
𝑝
=
𝑝
𝑖
,
𝑣
+
𝛾
𝑖
⋅
𝑝
𝑖
,
𝑣
(with probability 
𝑟
𝑖
)
p=p
i,v
	​

(normal)orp=p
i,v
	​

+γ
i
	​

⋅p
i,v
	​

(with probability r
i
	​

)



Availability & Setup Losses:

Stations can have downtime schedules to represent availability.

Variant changeovers can add setup times via Setup Matrix.
