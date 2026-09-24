# PyTorch 2.14 Release Live Q&A

来源：[https://www.youtube.com/watch?v=pRjbQa2wHB4](https://www.youtube.com/watch?v=pRjbQa2wHB4)

## 简介

PyTorch 2.14 introduces updates across compilation, distributed training, performance, dynamic shapes, Apple Silicon, and accelerator platforms. Highlights include NVGEMM, a CuTeDSL-generated GEMM backend for Inductor; the new nccl2 backend for PyTorch Distributed; fault-tolerant process-group reconfiguration in c10d; native linear algebra on Apple Silicon; and declarative dynamic shapes with @dynamic_spec.

Bring your questions about the release to our live Q&A. Andrey Talman (Meta), Nikkita Shulga (Thinking Machines Lab), Joe Spisak (Reflection AI), and Chris Gottbrath (Gottbrath Tech, moderator) will share an overview of PyTorch 2.14 and answer community questions about PyTorch and the new capabilities in the release.

Topics will include:

- NVGEMM and CuTeDSL-generated CUTLASS kernels in Inductor
- The new nccl2 backend for PyTorch Distributed
- Fault-tolerant collectives and process-group reconfiguration in c10d
- Native linear algebra and additional Metal kernel improvements on Apple Silicon
- torch.switch and CUDA graph capture for torch.while_loop
- Declarative dynamic shapes with @dynamic_spec
- Experimental torch.compile support for complex-valued tensors
- Expanded ROCm, Intel XPU, and NVIDIA platform support

PyTorch 2.14 includes 2,995 commits from 487 contributors since PyTorch 2.13. The release includes work across compilation, distributed communication, device support, and accelerator platforms. PyTorch Conference North America 2026 takes place October 20–21 in San Jose, with sessions spanning compiler and runtime work, distributed communication, device portability, release engineering, CI, observability, accelerator integration, contributor infrastructure, and more. Explore PyTorch Conference North America 2026.

## 字幕

6.710–6.720  Okay. Uh hello everyone. Um and welcome
6.720–10.150  Okay. Uh hello everyone. Um and welcome to our PyTorch uh 2.14
10.150–10.160  to our PyTorch uh 2.14
10.160–15.030  to our PyTorch uh 2.14 uh release live uh Q&A. Um uh let me
15.030–15.040  uh release live uh Q&A. Um uh let me
15.040–17.269  uh release live uh Q&A. Um uh let me scroll back up to my uh my comments
17.269–17.279  scroll back up to my uh my comments
17.279–20.390  scroll back up to my uh my comments here. Um this session will start with uh
20.390–20.400  here. Um this session will start with uh
20.400–22.950  here. Um this session will start with uh with introducing our experts um and then
22.950–22.960  with introducing our experts um and then
22.960–24.550  with introducing our experts um and then a short overview of the key updates in
24.550–24.560  a short overview of the key updates in
24.560–28.150  a short overview of the key updates in PyTorch uh 21.14 followed by um the fun
28.150–28.160  PyTorch uh 21.14 followed by um the fun
28.160–30.710  PyTorch uh 21.14 followed by um the fun part which is an open Q&A covering your
30.710–30.720  part which is an open Q&A covering your
30.720–33.030  part which is an open Q&A covering your questions about this release. Um first
33.030–33.040  questions about this release. Um first
33.040–34.389  questions about this release. Um first I'll introduce myself. I'm I'm Chris
34.389–34.399  I'll introduce myself. I'm I'm Chris
34.399–35.830  I'll introduce myself. I'm I'm Chris Scott. I'm your moderator for today's
35.830–35.840  Scott. I'm your moderator for today's
35.840–37.030  Scott. I'm your moderator for today's webinar. I've been involved in the
37.030–37.040  webinar. I've been involved in the
37.040–39.430  webinar. I've been involved in the PyTorch community for about eight years
39.430–39.440  PyTorch community for about eight years
39.440–41.190  PyTorch community for about eight years focusing on focusing on product and
41.190–41.200  focusing on focusing on product and
41.200–43.830  focusing on focusing on product and marketing uh type angles. Um these days
43.830–43.840  marketing uh type angles. Um these days
43.840–45.670  marketing uh type angles. Um these days I run a consulting practice uh focusing
45.670–45.680  I run a consulting practice uh focusing
45.680–48.229  I run a consulting practice uh focusing on emerging AI technologies and I'm
48.229–48.239  on emerging AI technologies and I'm
48.239–50.069  on emerging AI technologies and I'm delighted to welcome a panel of three
50.069–50.079  delighted to welcome a panel of three
50.079–52.389  delighted to welcome a panel of three absolute PyTorch experts. We have Andre
52.389–52.399  absolute PyTorch experts. We have Andre
52.399–55.990  absolute PyTorch experts. We have Andre Talman uh Joe Spizac and Nikita Schulga.
55.990–56.000  Talman uh Joe Spizac and Nikita Schulga.
56.000–57.430  Talman uh Joe Spizac and Nikita Schulga. Uh and they're here to answer your
57.430–57.440  Uh and they're here to answer your
57.440–59.349  Uh and they're here to answer your questions about PyTorch and generally to
59.349–59.359  questions about PyTorch and generally to
59.359–62.630  questions about PyTorch and generally to you know spread the word. Um so Andre is
62.630–62.640  you know spread the word. Um so Andre is
62.640–64.310  you know spread the word. Um so Andre is a release engineer in the PyTorch Dev
64.310–64.320  a release engineer in the PyTorch Dev
64.320–67.109  a release engineer in the PyTorch Dev Info team here at Meta or at Meta. I'm
67.109–67.119  Info team here at Meta or at Meta. I'm
67.119–69.109  Info team here at Meta or at Meta. I'm sorry. I used to be at Meta so that just
69.109–69.119  sorry. I used to be at Meta so that just
69.119–70.469  sorry. I used to be at Meta so that just rolls off my tongue. It's no longer
70.469–70.479  rolls off my tongue. It's no longer
70.479–73.830  rolls off my tongue. It's no longer accurate. at Meta. Uh he drives uh the
73.830–73.840  accurate. at Meta. Uh he drives uh the
73.840–75.429  accurate. at Meta. Uh he drives uh the release process and keeps uh the
75.429–75.439  release process and keeps uh the
75.439–76.789  release process and keeps uh the continuous integration and continuous
76.789–76.799  continuous integration and continuous
76.799–79.109  continuous integration and continuous development infrastructure humming. Uh
79.109–79.119  development infrastructure humming. Uh
79.119–83.109  development infrastructure humming. Uh Joe um is the uh VP of product uh and
83.109–83.119  Joe um is the uh VP of product uh and
83.119–85.910  Joe um is the uh VP of product uh and head of open source at Reflection AI. Uh
85.910–85.920  head of open source at Reflection AI. Uh
85.920–87.749  head of open source at Reflection AI. Uh he's one of the PyTorch core maintainers
87.749–87.759  he's one of the PyTorch core maintainers
87.759–89.590  he's one of the PyTorch core maintainers and among the strongest advocates that I
89.590–89.600  and among the strongest advocates that I
89.600–91.670  and among the strongest advocates that I know of for open source, open models and
91.670–91.680  know of for open source, open models and
91.680–93.429  know of for open source, open models and open infrastructure across the AI
93.429–93.439  open infrastructure across the AI
93.439–95.910  open infrastructure across the AI community. Uh and always a fun uh fun
95.910–95.920  community. Uh and always a fun uh fun
95.920–98.230  community. Uh and always a fun uh fun person to to hear from. Nikita is a
98.230–98.240  person to to hear from. Nikita is a
98.240–100.789  person to to hear from. Nikita is a PyTorch core maintainer uh and a key
100.789–100.799  PyTorch core maintainer uh and a key
100.799–102.870  PyTorch core maintainer uh and a key contributor and reviewer really active
102.870–102.880  contributor and reviewer really active
102.880–106.069  contributor and reviewer really active across just a bunch of areas uh in in
106.069–106.079  across just a bunch of areas uh in in
106.079–108.069  across just a bunch of areas uh in in the core area of PyTorch. So you
108.069–108.079  the core area of PyTorch. So you
108.079–109.590  the core area of PyTorch. So you probably see his name on on lots of
109.590–109.600  probably see his name on on lots of
109.600–112.870  probably see his name on on lots of things. Um I'll give a brief overview
112.870–112.880  things. Um I'll give a brief overview
112.880–114.550  things. Um I'll give a brief overview now in a in a moment here about the
114.550–114.560  now in a in a moment here about the
114.560–116.950  now in a in a moment here about the release uh for a few minutes and I'm
116.950–116.960  release uh for a few minutes and I'm
116.960–118.550  release uh for a few minutes and I'm would encourage you if you're in the in
118.550–118.560  would encourage you if you're in the in
118.560–120.550  would encourage you if you're in the in the audience uh to to use that time to
120.550–120.560  the audience uh to to use that time to
120.560–122.389  the audience uh to to use that time to think about what questions you might
122.389–122.399  think about what questions you might
122.399–125.109  think about what questions you might have um for our experts. uh you can put
125.109–125.119  have um for our experts. uh you can put
125.119–126.870  have um for our experts. uh you can put those in LinkedIn and and the and the
126.870–126.880  those in LinkedIn and and the and the
126.880–129.430  those in LinkedIn and and the and the YouTube tools. Uh we'll uh we'll cue
129.430–129.440  YouTube tools. Uh we'll uh we'll cue
129.440–132.309  YouTube tools. Uh we'll uh we'll cue them up for our experts to answer. We
132.309–132.319  them up for our experts to answer. We
132.319–133.750  them up for our experts to answer. We will we'll also have a few seed
133.750–133.760  will we'll also have a few seed
133.760–136.470  will we'll also have a few seed questions up uh on our you know prepared
136.470–136.480  questions up uh on our you know prepared
136.480–137.750  questions up uh on our you know prepared uh based on ongoing community
137.750–137.760  uh based on ongoing community
137.760–139.510  uh based on ongoing community discussions which you can join by uh
139.510–139.520  discussions which you can join by uh
139.520–141.990  discussions which you can join by uh looking them up in uh discuss.pytor.org
141.990–142.000  looking them up in uh discuss.pytor.org
142.000–144.630  looking them up in uh discuss.pytor.org and dev discuss.pytor.org or hopping on
144.630–144.640  and dev discuss.pytor.org or hopping on
144.640–147.270  and dev discuss.pytor.org or hopping on PyTorch Slack. Um those are just here to
147.270–147.280  PyTorch Slack. Um those are just here to
147.280–148.790  PyTorch Slack. Um those are just here to to kind of keep things flowing and make
148.790–148.800  to kind of keep things flowing and make
148.800–150.229  to kind of keep things flowing and make sure we touch on some important topics
150.229–150.239  sure we touch on some important topics
150.239–151.910  sure we touch on some important topics but it's more fun really to answer the
151.910–151.920  but it's more fun really to answer the
151.920–152.790  but it's more fun really to answer the questions that come in from the
152.790–152.800  questions that come in from the
152.800–155.670  questions that come in from the community. So, please don't be shy.
155.670–155.680  community. So, please don't be shy.
155.680–158.070  community. So, please don't be shy. Okay, so an overview of the release. Uh,
158.070–158.080  Okay, so an overview of the release. Uh,
158.080–159.350  Okay, so an overview of the release. Uh, this actually seems to me to be a pretty
159.350–159.360  this actually seems to me to be a pretty
159.360–160.630  this actually seems to me to be a pretty significant release despite, you know,
160.630–160.640  significant release despite, you know,
160.640–163.110  significant release despite, you know, the the numbering 21.14 makes it seem
163.110–163.120  the the numbering 21.14 makes it seem
163.120–164.790  the the numbering 21.14 makes it seem small, but it's not really, I don't
164.790–164.800  small, but it's not really, I don't
164.800–167.830  small, but it's not really, I don't think, at all. Um, uh, it really moves
167.830–167.840  think, at all. Um, uh, it really moves
167.840–170.150  think, at all. Um, uh, it really moves the needle on like compilation, uh, with
170.150–170.160  the needle on like compilation, uh, with
170.160–172.390  the needle on like compilation, uh, with some new concepts in compilation, fault
172.390–172.400  some new concepts in compilation, fault
172.400–175.190  some new concepts in compilation, fault tolerance, uh, and, uh, platform
175.190–175.200  tolerance, uh, and, uh, platform
175.200–177.270  tolerance, uh, and, uh, platform support. As always, it's a huge
177.270–177.280  support. As always, it's a huge
177.280–179.990  support. As always, it's a huge community left with almost 3,000 commits
179.990–180.000  community left with almost 3,000 commits
180.000–182.309  community left with almost 3,000 commits from almost 500 uh different uh
182.309–182.319  from almost 500 uh different uh
182.319–185.190  from almost 500 uh different uh contributors, 487 this year or this
185.190–185.200  contributors, 487 this year or this
185.200–187.750  contributors, 487 this year or this time. Um they made improvements all
187.750–187.760  time. Um they made improvements all
187.760–189.190  time. Um they made improvements all across PyTorch and I encourage folks to
189.190–189.200  across PyTorch and I encourage folks to
189.200–190.790  across PyTorch and I encourage folks to look at the release release blog and the
190.790–190.800  look at the release release blog and the
190.800–193.270  look at the release release blog and the release notes for uh for more detail to
193.270–193.280  release notes for uh for more detail to
193.280–195.750  release notes for uh for more detail to supplement this uh this Q&A. A few of
195.750–195.760  supplement this uh this Q&A. A few of
195.760–197.750  supplement this uh this Q&A. A few of the most spec uh most significant
197.750–197.760  the most spec uh most significant
197.760–201.589  the most spec uh most significant changes this time around are uh NV gym
201.589–201.599  changes this time around are uh NV gym
201.599–204.790  changes this time around are uh NV gym from Nvidia brings cute DSL uh generated
204.790–204.800  from Nvidia brings cute DSL uh generated
204.800–206.949  from Nvidia brings cute DSL uh generated cutless kernels to inductor. So this is
206.949–206.959  cutless kernels to inductor. So this is
206.959–209.830  cutless kernels to inductor. So this is like a performance uh improvement uh for
209.830–209.840  like a performance uh improvement uh for
209.840–213.190  like a performance uh improvement uh for uh modern uh GPUs uh with epilog fusion
213.190–213.200  uh modern uh GPUs uh with epilog fusion
213.200–217.430  uh modern uh GPUs uh with epilog fusion scaled uh NV FP4 gems and grouped uh
217.430–217.440  scaled uh NV FP4 gems and grouped uh
217.440–220.070  scaled uh NV FP4 gems and grouped uh reduction epilogues autotuned alongside
220.070–220.080  reduction epilogues autotuned alongside
220.080–222.869  reduction epilogues autotuned alongside Triton and ATIN.
222.869–222.879  Triton and ATIN.
222.879–225.670  Triton and ATIN. We have a preview of our uh some really
225.670–225.680  We have a preview of our uh some really
225.680–228.070  We have a preview of our uh some really deep changes in our in our um uh
228.070–228.080  deep changes in our in our um uh
228.080–230.470  deep changes in our in our um uh distributed backend. So rewritten nickel
230.470–230.480  distributed backend. So rewritten nickel
230.480–232.949  distributed backend. So rewritten nickel backend for PyTorch. This is ported from
232.949–232.959  backend for PyTorch. This is ported from
232.959–234.789  backend for PyTorch. This is ported from sort of a an area where we've been doing
234.789–234.799  sort of a an area where we've been doing
234.799–236.949  sort of a an area where we've been doing some experimentation called torchcoms
236.949–236.959  some experimentation called torchcoms
236.959–238.949  some experimentation called torchcoms implementing uh the full collective
238.949–238.959  implementing uh the full collective
238.959–241.670  implementing uh the full collective contract uh with non-blocking
241.670–241.680  contract uh with non-blocking
241.680–243.830  contract uh with non-blocking communication communicators and advanced
243.830–243.840  communication communicators and advanced
243.840–245.830  communication communicators and advanced features such as fault tolerance and
245.830–245.840  features such as fault tolerance and
245.840–247.910  features such as fault tolerance and windows uh designed as drag and drop
247.910–247.920  windows uh designed as drag and drop
247.920–250.390  windows uh designed as drag and drop replacements for existing nickel C10
250.390–250.400  replacements for existing nickel C10
250.400–253.110  replacements for existing nickel C10 backend. Um, so this I think is really
253.110–253.120  backend. Um, so this I think is really
253.120–254.710  backend. Um, so this I think is really uh the fault tolerance I think is going
254.710–254.720  uh the fault tolerance I think is going
254.720–256.710  uh the fault tolerance I think is going to be is going to be a big uh thing as
256.710–256.720  to be is going to be a big uh thing as
256.720–258.469  to be is going to be a big uh thing as we uh go forward and I think this is
258.469–258.479  we uh go forward and I think this is
258.479–261.189  we uh go forward and I think this is really laying really strong uh uh
261.189–261.199  really laying really strong uh uh
261.199–264.070  really laying really strong uh uh foundations for that. Uh fault tolerance
264.070–264.080  foundations for that. Uh fault tolerance
264.080–265.990  foundations for that. Uh fault tolerance also becomes a first class citizen in
265.990–266.000  also becomes a first class citizen in
266.000–269.670  also becomes a first class citizen in the C10 uh D uh with in place process
269.670–269.680  the C10 uh D uh with in place process
269.680–271.510  the C10 uh D uh with in place process group reconfiguration, one-sided RMA
271.510–271.520  group reconfiguration, one-sided RMA
271.520–273.430  group reconfiguration, one-sided RMA windows and a flight recorder that works
273.430–273.440  windows and a flight recorder that works
273.440–275.270  windows and a flight recorder that works for any backend rather than only for
275.270–275.280  for any backend rather than only for
275.280–277.990  for any backend rather than only for nickel. Uh Apple silicon also got a lot
277.990–278.000  nickel. Uh Apple silicon also got a lot
278.000–279.990  nickel. Uh Apple silicon also got a lot more u functionality with more native
279.990–280.000  more u functionality with more native
280.000–283.350  more u functionality with more native linear uh algebra operators including ig
283.350–283.360  linear uh algebra operators including ig
283.360–286.230  linear uh algebra operators including ig uh QR and Cholski alongside uh fast
286.230–286.240  uh QR and Cholski alongside uh fast
286.240–288.870  uh QR and Cholski alongside uh fast production operations and further uh uh
288.870–288.880  production operations and further uh uh
288.880–291.510  production operations and further uh uh MPS graph to metal uh kernel migrations.
291.510–291.520  MPS graph to metal uh kernel migrations.
291.520–293.830  MPS graph to metal uh kernel migrations. So performance improvements uh in many
293.830–293.840  So performance improvements uh in many
293.840–296.870  So performance improvements uh in many places on Apple um torch switch
296.870–296.880  places on Apple um torch switch
296.880–299.430  places on Apple um torch switch generalize uh the torch cond to
299.430–299.440  generalize uh the torch cond to
299.440–300.870  generalize uh the torch cond to multi-way branching. So this is like a
300.870–300.880  multi-way branching. So this is like a
300.880–302.390  multi-way branching. So this is like a language improvement in the way you can
302.390–302.400  language improvement in the way you can
302.400–306.390  language improvement in the way you can think about um uh uh expressing uh
306.390–306.400  think about um uh uh expressing uh
306.400–308.310  think about um uh uh expressing uh things that used to cause graph breaks
308.310–308.320  things that used to cause graph breaks
308.320–311.909  things that used to cause graph breaks in PyTorch. Uh with torch while loop uh
311.909–311.919  in PyTorch. Uh with torch while loop uh
311.919–313.670  in PyTorch. Uh with torch while loop uh also can now be captured in a CUDA
313.670–313.680  also can now be captured in a CUDA
313.680–316.950  also can now be captured in a CUDA graph. Um declarative dynamic shapes uh
316.950–316.960  graph. Um declarative dynamic shapes uh
316.960–319.110  graph. Um declarative dynamic shapes uh are also uh added. This is another sort
319.110–319.120  are also uh added. This is another sort
319.120–321.110  are also uh added. This is another sort of language feature. There's now a
321.110–321.120  of language feature. There's now a
321.120–324.070  of language feature. There's now a dynamic spec uh uh decorator at dynamic
324.070–324.080  dynamic spec uh uh decorator at dynamic
324.080–325.990  dynamic spec uh uh decorator at dynamic spec uh which is shared across torch
325.990–326.000  spec uh which is shared across torch
326.000–328.790  spec uh which is shared across torch compile torch export and make.fx FX
328.790–328.800  compile torch export and make.fx FX
328.800–330.870  compile torch export and make.fx FX experimental torch compiler uh support
330.870–330.880  experimental torch compiler uh support
330.880–332.870  experimental torch compiler uh support is also available for complex value
332.870–332.880  is also available for complex value
332.880–335.830  is also available for complex value tensors. Uh this is an opt-in thing uh
335.830–335.840  tensors. Uh this is an opt-in thing uh
335.840–338.310  tensors. Uh this is an opt-in thing uh which decomposes uh supported complex
338.310–338.320  which decomposes uh supported complex
338.320–339.670  which decomposes uh supported complex operations into real and imaginary
339.670–339.680  operations into real and imaginary
339.680–341.830  operations into real and imaginary computations enabling compiler backends
341.830–341.840  computations enabling compiler backends
341.840–344.469  computations enabling compiler backends to optimize complex number workloads
344.469–344.479  to optimize complex number workloads
344.479–346.070  to optimize complex number workloads which are super important for science
346.070–346.080  which are super important for science
346.080–347.749  which are super important for science and I think uh have been hard to do in
347.749–347.759  and I think uh have been hard to do in
347.759–350.230  and I think uh have been hard to do in PyTorch for a long time. Um broader
350.230–350.240  PyTorch for a long time. Um broader
350.240–352.790  PyTorch for a long time. Um broader platform support uh is you know we also
352.790–352.800  platform support uh is you know we also
352.800–353.909  platform support uh is you know we also made improvements across a bunch of
353.909–353.919  made improvements across a bunch of
353.919–355.749  made improvements across a bunch of different platforms including Rockcom.
355.749–355.759  different platforms including Rockcom.
355.759–359.350  different platforms including Rockcom. There's now 7.14 wheels uh produced um
359.350–359.360  There's now 7.14 wheels uh produced um
359.360–363.830  There's now 7.14 wheels uh produced um from the rock pip uh SDK and Intel
363.830–363.840  from the rock pip uh SDK and Intel
363.840–366.230  from the rock pip uh SDK and Intel adding uh native graph capture for the
366.230–366.240  adding uh native graph capture for the
366.240–369.830  adding uh native graph capture for the XPU and inductor uh target inductor
369.830–369.840  XPU and inductor uh target inductor
369.840–372.870  XPU and inductor uh target inductor targeting uh inductor uh functionality
372.870–372.880  targeting uh inductor uh functionality
372.880–374.790  targeting uh inductor uh functionality targeting the new Ruben platform from
374.790–374.800  targeting the new Ruben platform from
374.800–377.830  targeting the new Ruben platform from Nvidia. So that was a mouthful but
377.830–377.840  Nvidia. So that was a mouthful but
377.840–379.590  Nvidia. So that was a mouthful but that's the sort of like highlights from
379.590–379.600  that's the sort of like highlights from
379.600–381.590  that's the sort of like highlights from this particular release. Having set that
381.590–381.600  this particular release. Having set that
381.600–384.629  this particular release. Having set that stage let's move to questions. Um so at
384.629–384.639  stage let's move to questions. Um so at
384.639–386.790  stage let's move to questions. Um so at the first question I think that we might
386.790–386.800  the first question I think that we might
386.800–391.909  the first question I think that we might have
395.430–395.440  will be what CUDA version does PyTorch
395.440–397.830  will be what CUDA version does PyTorch 2.14 ship
397.830–397.840  2.14 ship
397.840–400.309  2.14 ship uh and which one do I get by default and
400.309–400.319  uh and which one do I get by default and
400.319–402.629  uh and which one do I get by default and what packaging changes for CUDA do we
402.629–402.639  what packaging changes for CUDA do we
402.639–406.094  what packaging changes for CUDA do we expect for the upcoming release? Andre
406.094–406.104  expect for the upcoming release? Andre
406.104–406.150  expect for the upcoming release? Andre >> [snorts]
406.150–406.160  >> [snorts]
406.160–408.710  >> [snorts] >> um yes so basically for the current
408.710–408.720  >> um yes so basically for the current
408.720–412.550  >> um yes so basically for the current 21.14 release uh we support exactly same
412.550–412.560  21.14 release uh we support exactly same
412.560–416.230  21.14 release uh we support exactly same cuda matrix as previous release 21.13 so
416.230–416.240  cuda matrix as previous release 21.13 so
416.240–422.309  cuda matrix as previous release 21.13 so it's cuda 12.6 kuda 13.0 and cuda 13.2
422.309–422.319  it's cuda 12.6 kuda 13.0 and cuda 13.2
422.319–426.469  it's cuda 12.6 kuda 13.0 and cuda 13.2 um cuda 13.0 is being published to pipy
426.469–426.479  um cuda 13.0 is being published to pipy
426.479–431.029  um cuda 13.0 is being published to pipy um cuda 13.2 two becomes stable. Um it
431.029–431.039  um cuda 13.2 two becomes stable. Um it
431.039–433.510  um cuda 13.2 two becomes stable. Um it for previous release it was prototype
433.510–433.520  for previous release it was prototype
433.520–435.350  for previous release it was prototype considered as prototype right now it's
435.350–435.360  considered as prototype right now it's
435.360–440.710  considered as prototype right now it's stable. Um so um for the next uh also
440.710–440.720  stable. Um so um for the next uh also
440.720–443.270  stable. Um so um for the next uh also for this release we included rocker
443.270–443.280  for this release we included rocker
443.280–446.629  for this release we included rocker modifications. So rocom 714 wheels are
446.629–446.639  modifications. So rocom 714 wheels are
446.639–448.150  modifications. So rocom 714 wheels are published
448.150–448.160  published
448.160–451.589  published um right now build with uh the rock uh
451.589–451.599  um right now build with uh the rock uh
451.599–453.510  um right now build with uh the rock uh pip SDK.
453.510–453.520  pip SDK.
453.520–457.029  pip SDK. Um so for the next release 2.15 it's I
457.029–457.039  Um so for the next release 2.15 it's I
457.039–458.710  Um so for the next release 2.15 it's I think where the lot of changes are
458.710–458.720  think where the lot of changes are
458.720–464.150  think where the lot of changes are coming um first of all 2 um 14 release
464.150–464.160  coming um first of all 2 um 14 release
464.160–466.469  coming um first of all 2 um 14 release was the last release that we released
466.469–466.479  was the last release that we released
466.479–470.070  was the last release that we released CUDA 12.x wheels so 12.6 six for the
470.070–470.080  CUDA 12.x wheels so 12.6 six for the
470.080–472.710  CUDA 12.x wheels so 12.6 six for the 2.15 release. We will uh remove those
472.710–472.720  2.15 release. We will uh remove those
472.720–475.589  2.15 release. We will uh remove those wheels. There will no longer be shipped
475.589–475.599  wheels. There will no longer be shipped
475.599–478.869  wheels. There will no longer be shipped with PyTorch. However, you are able you
478.869–478.879  with PyTorch. However, you are able you
478.879–481.350  with PyTorch. However, you are able you will still be able to compile
481.350–481.360  will still be able to compile
481.360–483.510  will still be able to compile with those manually
483.510–483.520  with those manually
483.520–486.550  with those manually um because and we are keeping one CUDA
486.550–486.560  um because and we are keeping one CUDA
486.560–489.510  um because and we are keeping one CUDA 12.6 CI built on our system. So we are
489.510–489.520  12.6 CI built on our system. So we are
489.520–492.230  12.6 CI built on our system. So we are ensuring that it's still the case. So
492.230–492.240  ensuring that it's still the case. So
492.240–495.510  ensuring that it's still the case. So there is no regression in this case. Um
495.510–495.520  there is no regression in this case. Um
495.520–498.950  there is no regression in this case. Um with CUDA 12.6 six support um uh that
498.950–498.960  with CUDA 12.6 six support um uh that
498.960–501.350  with CUDA 12.6 six support um uh that we're removing we're no longer be
501.350–501.360  we're removing we're no longer be
501.360–503.830  we're removing we're no longer be supporting pre-built wheels for Maxwell,
503.830–503.840  supporting pre-built wheels for Maxwell,
503.840–507.589  supporting pre-built wheels for Maxwell, Pascal and VA architectures.
507.589–507.599  Pascal and VA architectures.
507.599–511.990  Pascal and VA architectures. Um so the next part uh it's CUDA 13.2
511.990–512.000  Um so the next part uh it's CUDA 13.2
512.000–514.469  Um so the next part uh it's CUDA 13.2 will be published as default CUDA
514.469–514.479  will be published as default CUDA
514.479–518.230  will be published as default CUDA version on PIP and we will introduce
518.230–518.240  version on PIP and we will introduce
518.240–522.949  version on PIP and we will introduce CUDA 13.4 four as a prototype uh CUDA
522.949–522.959  CUDA 13.4 four as a prototype uh CUDA
522.959–525.269  CUDA 13.4 four as a prototype uh CUDA version with support for ver Rubin
525.269–525.279  version with support for ver Rubin
525.279–527.990  version with support for ver Rubin architecture. Um I think this covers
527.990–528.000  architecture. Um I think this covers
528.000–530.230  architecture. Um I think this covers pretty much it in terms of cuda and a
530.230–530.240  pretty much it in terms of cuda and a
530.240–532.230  pretty much it in terms of cuda and a little bit of rocom. Please
532.230–532.240  little bit of rocom. Please
532.240–534.550  little bit of rocom. Please >> if I can make a small comment there's a
534.550–534.560  >> if I can make a small comment there's a
534.560–537.269  >> if I can make a small comment there's a like a lot of numbers lot of versions
537.269–537.279  like a lot of numbers lot of versions
537.279–540.310  like a lot of numbers lot of versions please track the pytor issues and dev
540.310–540.320  please track the pytor issues and dev
540.320–542.150  please track the pytor issues and dev discuss where we publish all those
542.150–542.160  discuss where we publish all those
542.160–544.150  discuss where we publish all those versions because it's they hard to
544.150–544.160  versions because it's they hard to
544.160–545.990  versions because it's they hard to remember and sometimes we have to pivot
545.990–546.000  remember and sometimes we have to pivot
546.000–547.509  remember and sometimes we have to pivot depending on some new hardware
547.509–547.519  depending on some new hardware
547.519–549.670  depending on some new hardware capabilities or whatever but that's the
549.670–549.680  capabilities or whatever but that's the
549.680–552.150  capabilities or whatever but that's the current plans. Thank you.
552.150–552.160  current plans. Thank you.
552.160–553.990  current plans. Thank you. Yeah, and just to add, there's a release
553.990–554.000  Yeah, and just to add, there's a release
554.000–556.550  Yeah, and just to add, there's a release MD file in the PyTorch PyTorch where the
556.550–556.560  MD file in the PyTorch PyTorch where the
556.560–558.550  MD file in the PyTorch PyTorch where the current matrix is published and yeah,
558.550–558.560  current matrix is published and yeah,
558.560–560.230  current matrix is published and yeah, you can go ahead and take a look there
560.230–560.240  you can go ahead and take a look there
560.240–565.509  you can go ahead and take a look there and ask any questions.
567.990–568.000  >> Okay. Um, do we have a next question
568.000–571.811  >> Okay. Um, do we have a next question queued up?
572.710–572.720  [snorts]
572.720–575.190  [snorts] Okay. Um, thank you, uh, Michael for
575.190–575.200  Okay. Um, thank you, uh, Michael for
575.200–577.110  Okay. Um, thank you, uh, Michael for asking this question. With Agentic RL
577.110–577.120  asking this question. With Agentic RL
577.120–579.350  asking this question. With Agentic RL becoming a bigger part of training, um
579.350–579.360  becoming a bigger part of training, um
579.360–581.110  becoming a bigger part of training, um are you all seeing these workloads
581.110–581.120  are you all seeing these workloads
581.120–582.070  are you all seeing these workloads create different infrastructure
582.070–582.080  create different infrastructure
582.080–583.350  create different infrastructure challenges than traditional model
583.350–583.360  challenges than traditional model
583.360–585.829  challenges than traditional model training? Anything PyTorch needs still
585.829–585.839  training? Anything PyTorch needs still
585.839–587.430  training? Anything PyTorch needs still needs to get better at there? I'm
587.430–587.440  needs to get better at there? I'm
587.440–589.269  needs to get better at there? I'm actually going to um direct this one to
589.269–589.279  actually going to um direct this one to
589.279–591.509  actually going to um direct this one to Joe first and then Andre and Nikita, you
591.509–591.519  Joe first and then Andre and Nikita, you
591.519–593.910  Joe first and then Andre and Nikita, you guys can jump in after. Um Joe, I know
593.910–593.920  guys can jump in after. Um Joe, I know
593.920–596.870  guys can jump in after. Um Joe, I know you care a lot about agentic RL. So
596.870–596.880  you care a lot about agentic RL. So
596.880–598.150  you care a lot about agentic RL. So >> yeah,
598.150–598.160  >> yeah,
598.160–600.550  >> yeah, >> I mean I think um you know if you look
600.550–600.560  >> I mean I think um you know if you look
600.560–602.710  >> I mean I think um you know if you look at it's it's pretty interesting the the
602.710–602.720  at it's it's pretty interesting the the
602.720–604.150  at it's it's pretty interesting the the models coming out these days are coming
604.150–604.160  models coming out these days are coming
604.160–606.550  models coming out these days are coming out fast and furious and I think there
606.550–606.560  out fast and furious and I think there
606.560–608.310  out fast and furious and I think there there was at least a for a while a
608.310–608.320  there was at least a for a while a
608.320–609.509  there was at least a for a while a perception that people are are
609.509–609.519  perception that people are are
609.519–611.509  perception that people are are pre-training models at an increased pace
611.509–611.519  pre-training models at an increased pace
611.519–612.949  pre-training models at an increased pace right but I think a lot of the gains
612.949–612.959  right but I think a lot of the gains
612.959–614.710  right but I think a lot of the gains that that we've seen actually you know
614.710–614.720  that that we've seen actually you know
614.720–616.630  that that we've seen actually you know some of the Chinese models for example
616.630–616.640  some of the Chinese models for example
616.640–618.470  some of the Chinese models for example that are coming out and and versioned
618.470–618.480  that are coming out and and versioned
618.480–620.550  that are coming out and and versioned really fast and the Quen models a lot of
620.550–620.560  really fast and the Quen models a lot of
620.560–623.269  really fast and the Quen models a lot of it is actually due to RL uh and you know
623.269–623.279  it is actually due to RL uh and you know
623.279–624.710  it is actually due to RL uh and you know I think one of the things that we've
624.710–624.720  I think one of the things that we've
624.720–626.790  I think one of the things that we've learned over the is you know if you
626.790–626.800  learned over the is you know if you
626.800–629.509  learned over the is you know if you build a really strong pre-trained base
629.509–629.519  build a really strong pre-trained base
629.519–632.310  build a really strong pre-trained base um to build on you can actually RL over
632.310–632.320  um to build on you can actually RL over
632.320–633.509  um to build on you can actually RL over it and you can actually throw more
633.509–633.519  it and you can actually throw more
633.519–635.350  it and you can actually throw more compute on it and I think the deepseek
635.350–635.360  compute on it and I think the deepseek
635.360–637.110  compute on it and I think the deepseek paper actually taught us a lot of
637.110–637.120  paper actually taught us a lot of
637.120–639.110  paper actually taught us a lot of lessons that kind of confirms for the
639.110–639.120  lessons that kind of confirms for the
639.120–641.990  lessons that kind of confirms for the the bitter lesson pill people among us
641.990–642.000  the bitter lesson pill people among us
642.000–643.430  the bitter lesson pill people among us including myself that you know if you
643.430–643.440  including myself that you know if you
643.440–646.310  including myself that you know if you continue to throw more compute at stable
646.310–646.320  continue to throw more compute at stable
646.320–648.550  continue to throw more compute at stable stable RL using that word stable very
648.550–648.560  stable RL using that word stable very
648.560–651.590  stable RL using that word stable very very uh carefully purposefully uh then
651.590–651.600  very uh carefully purposefully uh then
651.600–653.350  very uh carefully purposefully uh then you can actually extend capabilities of
653.350–653.360  you can actually extend capabilities of
653.360–656.550  you can actually extend capabilities of models over time. And so I think like
656.550–656.560  models over time. And so I think like
656.560–659.350  models over time. And so I think like you know uh in the earlier days I guess
659.350–659.360  you know uh in the earlier days I guess
659.360–661.670  you know uh in the earlier days I guess of RL um and and we started this project
661.670–661.680  of RL um and and we started this project
661.680–664.150  of RL um and and we started this project called OpenM last year when I was
664.150–664.160  called OpenM last year when I was
664.160–666.230  called OpenM last year when I was actually almost one year ago exactly uh
666.230–666.240  actually almost one year ago exactly uh
666.240–668.710  actually almost one year ago exactly uh at the PyTorch conference we uh launched
668.710–668.720  at the PyTorch conference we uh launched
668.720–669.990  at the PyTorch conference we uh launched that project and it continues to grow
669.990–670.000  that project and it continues to grow
670.000–672.069  that project and it continues to grow and I think one the whole goal of it was
672.069–672.079  and I think one the whole goal of it was
672.079–673.910  and I think one the whole goal of it was to democratize the use of reinforcement
673.910–673.920  to democratize the use of reinforcement
673.920–676.389  to democratize the use of reinforcement learning um because it's it's a lot
676.389–676.399  learning um because it's it's a lot
676.399–678.389  learning um because it's it's a lot easier to kind of create an environment
678.389–678.399  easier to kind of create an environment
678.399–680.550  easier to kind of create an environment create your tasks um and then be able to
680.550–680.560  create your tasks um and then be able to
680.560–682.870  create your tasks um and then be able to to set up you know especially today with
682.870–682.880  to set up you know especially today with
682.880–685.110  to set up you know especially today with all the tools available Nemo RL
685.110–685.120  all the tools available Nemo RL
685.120–687.670  all the tools available Nemo RL TRL unsloth, you can actually uh use
687.670–687.680  TRL unsloth, you can actually uh use
687.680–690.710  TRL unsloth, you can actually uh use reinforcement learning um you know way
690.710–690.720  reinforcement learning um you know way
690.720–691.990  reinforcement learning um you know way easier than you could probably like five
691.990–692.000  easier than you could probably like five
692.000–694.069  easier than you could probably like five or 10 years ago in the earlier days um
694.069–694.079  or 10 years ago in the earlier days um
694.079–696.150  or 10 years ago in the earlier days um and actually add capabilities to your
696.150–696.160  and actually add capabilities to your
696.160–698.150  and actually add capabilities to your models bas basically be able to postrain
698.150–698.160  models bas basically be able to postrain
698.160–699.750  models bas basically be able to postrain using reinforcement learning for openw
699.750–699.760  using reinforcement learning for openw
699.760–701.190  using reinforcement learning for openw weight models and I think that's
701.190–701.200  weight models and I think that's
701.200–702.150  weight models and I think that's something we want to continue to
702.150–702.160  something we want to continue to
702.160–704.630  something we want to continue to obviously de democratize and and enable
704.630–704.640  obviously de democratize and and enable
704.640–706.630  obviously de democratize and and enable with developers um and so like Daniel
706.630–706.640  with developers um and so like Daniel
706.640–708.630  with developers um and so like Daniel from Unsloth continues to bear that that
708.630–708.640  from Unsloth continues to bear that that
708.640–710.949  from Unsloth continues to bear that that flag for many uh and obviously we
710.949–710.959  flag for many uh and obviously we
710.959–712.630  flag for many uh and obviously we contribute uh significantly to open
712.630–712.640  contribute uh significantly to open
712.640–714.470  contribute uh significantly to open project and that fits nicely into the
714.470–714.480  project and that fits nicely into the
714.480–716.790  project and that fits nicely into the PyTorch ecosystem and extends it as
716.790–716.800  PyTorch ecosystem and extends it as
716.800–718.870  PyTorch ecosystem and extends it as people use PyTorch and other tools to uh
718.870–718.880  people use PyTorch and other tools to uh
718.880–721.190  people use PyTorch and other tools to uh to to to customize their models for
721.190–721.200  to to to customize their models for
721.200–723.829  to to to customize their models for their tasks.
723.829–723.839  their tasks.
723.839–725.829  their tasks. >> But fundamentally, PyTorch has a lot of
725.829–725.839  >> But fundamentally, PyTorch has a lot of
725.839–728.389  >> But fundamentally, PyTorch has a lot of the things that that folks need. Um we
728.389–728.399  the things that that folks need. Um we
728.399–729.670  the things that that folks need. Um we need these sort of layered packages like
729.670–729.680  need these sort of layered packages like
729.680–731.590  need these sort of layered packages like OpenM but but PyTorch itself. Are there
731.590–731.600  OpenM but but PyTorch itself. Are there
731.600–734.150  OpenM but but PyTorch itself. Are there any changes that you see uh you see
734.150–734.160  any changes that you see uh you see
734.160–736.710  any changes that you see uh you see needed now or in the future in PyTorch?
736.710–736.720  needed now or in the future in PyTorch?
736.720–738.949  needed now or in the future in PyTorch? >> Well, I think
738.949–738.959  >> Well, I think
738.959–740.150  >> Well, I think Oh, sorry. Go for it.
740.150–740.160  Oh, sorry. Go for it.
740.160–741.750  Oh, sorry. Go for it. >> No, no, go ahead. I want to hear your
741.750–741.760  >> No, no, go ahead. I want to hear your
741.760–743.430  >> No, no, go ahead. I want to hear your point of view. on the infrastructure
743.430–743.440  point of view. on the infrastructure
743.440–746.150  point of view. on the infrastructure side like uh the nickel 2 and like the
746.150–746.160  side like uh the nickel 2 and like the
746.160–747.829  side like uh the nickel 2 and like the safer nickel is one of the things that
747.829–747.839  safer nickel is one of the things that
747.839–750.230  safer nickel is one of the things that is constantly painful for post training
750.230–750.240  is constantly painful for post training
750.240–752.310  is constantly painful for post training workflows when you have to restart. So
752.310–752.320  workflows when you have to restart. So
752.320–753.829  workflows when you have to restart. So that would be an infrastructure part but
753.829–753.839  that would be an infrastructure part but
753.839–755.590  that would be an infrastructure part but also we have as Joel Joe said a lot of
755.590–755.600  also we have as Joel Joe said a lot of
755.600–758.069  also we have as Joel Joe said a lot of great partnership. I encourage people to
758.069–758.079  great partnership. I encourage people to
758.079–761.110  great partnership. I encourage people to check out well u maybe not encourage but
761.110–761.120  check out well u maybe not encourage but
761.120–763.030  check out well u maybe not encourage but there are two frameworks that exist in
763.030–763.040  there are two frameworks that exist in
763.040–764.629  there are two frameworks that exist in the pytor side. There is a torch titan
764.629–764.639  the pytor side. There is a torch titan
764.639–767.269  the pytor side. There is a torch titan and torch forge that kind of give you
767.269–767.279  and torch forge that kind of give you
767.279–769.190  and torch forge that kind of give you some recipes on the post training and I
769.190–769.200  some recipes on the post training and I
769.200–770.550  some recipes on the post training and I think you should kind of direct your
770.550–770.560  think you should kind of direct your
770.560–772.230  think you should kind of direct your attention more towards those
772.230–772.240  attention more towards those
772.240–773.750  attention more towards those repositories when you're asking a
773.750–773.760  repositories when you're asking a
773.760–775.350  repositories when you're asking a specific questions about the frameworks
775.350–775.360  specific questions about the frameworks
775.360–777.910  specific questions about the frameworks rather about the like harness rather
777.910–777.920  rather about the like harness rather
777.920–780.629  rather about the like harness rather than like a training framework itself
780.629–780.639  than like a training framework itself
780.639–782.550  than like a training framework itself but support for MFP4 is also very
782.550–782.560  but support for MFP4 is also very
782.560–785.110  but support for MFP4 is also very important for post training pipelines so
785.110–785.120  important for post training pipelines so
785.120–786.470  important for post training pipelines so like all the infrastructures that is
786.470–786.480  like all the infrastructures that is
786.480–789.269  like all the infrastructures that is needed is there but Joe to continue
789.269–789.279  needed is there but Joe to continue
789.279–790.949  needed is there but Joe to continue >> no I I think you're spot on I think
790.949–790.959  >> no I I think you're spot on I think
790.959–793.110  >> no I I think you're spot on I think there you know infra I mean we're we're
793.110–793.120  there you know infra I mean we're we're
793.120–795.910  there you know infra I mean we're we're using a pretty large large amount of
795.910–795.920  using a pretty large large amount of
795.920–798.069  using a pretty large large amount of compute to do RL here in reflection. We
798.069–798.079  compute to do RL here in reflection. We
798.079–800.230  compute to do RL here in reflection. We obviously did a lot of RL meta. Um I
800.230–800.240  obviously did a lot of RL meta. Um I
800.240–801.990  obviously did a lot of RL meta. Um I think infrastructure is the challenge. I
801.990–802.000  think infrastructure is the challenge. I
802.000–803.190  think infrastructure is the challenge. I think there there's a lot of work I
803.190–803.200  think there there's a lot of work I
803.200–806.069  think there there's a lot of work I think to um you know to to scale it up
806.069–806.079  think to um you know to to scale it up
806.079–807.509  think to um you know to to scale it up to make it stable. I think I'm just
807.509–807.519  to make it stable. I think I'm just
807.519–809.670  to make it stable. I think I'm just looking at even like I'm I can't really
809.670–809.680  looking at even like I'm I can't really
809.680–810.710  looking at even like I'm I can't really use examples of what we're doing
810.710–810.720  use examples of what we're doing
810.720–811.829  use examples of what we're doing internally yet because we're still
811.829–811.839  internally yet because we're still
811.839–814.069  internally yet because we're still pretty salty. But like even when we were
814.069–814.079  pretty salty. But like even when we were
814.079–816.150  pretty salty. But like even when we were uh dealing with like fault tolerance in
816.150–816.160  uh dealing with like fault tolerance in
816.160–817.990  uh dealing with like fault tolerance in like the llama days, you know, most of
817.990–818.000  like the llama days, you know, most of
818.000–819.750  like the llama days, you know, most of our uh the faults that we found even in
819.750–819.760  our uh the faults that we found even in
819.760–821.509  our uh the faults that we found even in pre-training or even in post- training
821.509–821.519  pre-training or even in post- training
821.519–824.470  pre-training or even in post- training came from obviously GPUs failing over.
824.470–824.480  came from obviously GPUs failing over.
824.480–827.670  came from obviously GPUs failing over. um you know we had uh yeah we I think
827.670–827.680  um you know we had uh yeah we I think
827.680–829.590  um you know we had uh yeah we I think what something like 30% of of those
829.590–829.600  what something like 30% of of those
829.600–832.150  what something like 30% of of those failures or we had close to like 400 and
832.150–832.160  failures or we had close to like 400 and
832.160–834.629  failures or we had close to like 400 and something GPUs out of the 16,000 GPUs we
834.629–834.639  something GPUs out of the 16,000 GPUs we
834.639–836.550  something GPUs out of the 16,000 GPUs we were training on for the large model
836.550–836.560  were training on for the large model
836.560–837.910  were training on for the large model were failing. We wrote that into the
837.910–837.920  were failing. We wrote that into the
837.920–839.189  were failing. We wrote that into the paper. You can kind of look at the graph
839.189–839.199  paper. You can kind of look at the graph
839.199–841.269  paper. You can kind of look at the graph there. I think like infrastructure
841.269–841.279  there. I think like infrastructure
841.279–843.990  there. I think like infrastructure stability and um and fall tolerance and
843.990–844.000  stability and um and fall tolerance and
844.000–845.269  stability and um and fall tolerance and I think we'll get to that in some of the
845.269–845.279  I think we'll get to that in some of the
845.279–847.269  I think we'll get to that in some of the other questions but that's probably one
847.269–847.279  other questions but that's probably one
847.279–848.550  other questions but that's probably one of the biggest biggest things and I
848.550–848.560  of the biggest biggest things and I
848.560–849.750  of the biggest biggest things and I think that's one of the reasons why like
849.750–849.760  think that's one of the reasons why like
849.760–851.509  think that's one of the reasons why like Tinker is actually like pretty
851.509–851.519  Tinker is actually like pretty
851.519–853.189  Tinker is actually like pretty interesting from from thinking machines.
853.189–853.199  interesting from from thinking machines.
853.199–855.110  interesting from from thinking machines. It's like a, you know, not not to to
855.110–855.120  It's like a, you know, not not to to
855.120–857.509  It's like a, you know, not not to to plug the his product here, but like it
857.509–857.519  plug the his product here, but like it
857.519–859.910  plug the his product here, but like it it actually does um it's actually a
859.910–859.920  it actually does um it's actually a
859.920–862.069  it actually does um it's actually a really uh cool API that actually
862.069–862.079  really uh cool API that actually
862.079–863.910  really uh cool API that actually abstracts that away. And I think that's
863.910–863.920  abstracts that away. And I think that's
863.920–865.590  abstracts that away. And I think that's for those especially the academics I've
865.590–865.600  for those especially the academics I've
865.600–867.670  for those especially the academics I've talked to and researchers like that is
867.670–867.680  talked to and researchers like that is
867.680–869.430  talked to and researchers like that is by far the hardest part. And obviously
869.430–869.440  by far the hardest part. And obviously
869.440–871.030  by far the hardest part. And obviously it's powered by PyTorch which is really
871.030–871.040  it's powered by PyTorch which is really
871.040–873.350  it's powered by PyTorch which is really great. Um but that that is like the the
873.350–873.360  great. Um but that that is like the the
873.360–874.710  great. Um but that that is like the the thing that that needs to be solved for
874.710–874.720  thing that that needs to be solved for
874.720–876.069  thing that that needs to be solved for this to be something that's widespread
876.069–876.079  this to be something that's widespread
876.079–878.710  this to be something that's widespread used. So
878.710–878.720  used. So
878.720–880.550  used. So >> awesome. I think that's a pretty good
880.550–880.560  >> awesome. I think that's a pretty good
880.560–883.189  >> awesome. I think that's a pretty good pretty solid answer to that question.
883.189–883.199  pretty solid answer to that question.
883.199–887.110  pretty solid answer to that question. Um, next up, um, why didn't you add LLM?
887.110–887.120  Um, next up, um, why didn't you add LLM?
887.120–890.150  Um, next up, um, why didn't you add LLM? Why don't didn't you add LLM support
890.150–890.160  Why don't didn't you add LLM support
890.160–892.230  Why don't didn't you add LLM support into the UI so people can talk to an
892.230–892.240  into the UI so people can talk to an
892.240–893.670  into the UI so people can talk to an agent instead of reading through the
893.670–893.680  agent instead of reading through the
893.680–896.150  agent instead of reading through the documentation?
896.150–896.160  documentation?
896.160–899.315  documentation? Uh, who wants to answer this? Uh,
899.315–899.325  Uh, who wants to answer this? Uh,
899.325–899.829  Uh, who wants to answer this? Uh, [laughter]
899.829–899.839  [laughter]
899.839–901.509  [laughter] >> I'm not sure what UI we're talking
901.509–901.519  >> I'm not sure what UI we're talking
901.519–903.110  >> I'm not sure what UI we're talking about, but yes, like you can ask
903.110–903.120  about, but yes, like you can ask
903.120–905.430  about, but yes, like you can ask questions to an agent uh on the
905.430–905.440  questions to an agent uh on the
905.440–907.910  questions to an agent uh on the pytor.org, work but there's also kind of
907.910–907.920  pytor.org, work but there's also kind of
907.920–910.389  pytor.org, work but there's also kind of a pragmatic approach that um like
910.389–910.399  a pragmatic approach that um like
910.399–913.350  a pragmatic approach that um like running LM for public consumption is not
913.350–913.360  running LM for public consumption is not
913.360–915.670  running LM for public consumption is not cheap and via foundation right so there
915.670–915.680  cheap and via foundation right so there
915.680–918.230  cheap and via foundation right so there are lots of open LLMs that are very very
918.230–918.240  are lots of open LLMs that are very very
918.240–919.670  are lots of open LLMs that are very very good about answering questions about
919.670–919.680  good about answering questions about
919.680–921.990  good about answering questions about PyTorch documentation and we partnered
921.990–922.000  PyTorch documentation and we partnered
922.000–923.990  PyTorch documentation and we partnered with one of the providers early on so
923.990–924.000  with one of the providers early on so
924.000–926.710  with one of the providers early on so you can ask PyTorch questions that I
926.710–926.720  you can ask PyTorch questions that I
926.720–928.310  you can ask PyTorch questions that I don't remember job probably have more
928.310–928.320  don't remember job probably have more
928.320–929.910  don't remember job probably have more upto-ate details about the partnership
929.910–929.920  upto-ate details about the partnership
929.920–934.150  upto-ate details about the partnership like who is the provider but yeah
934.150–934.160  like who is the provider but yeah
934.160–936.629  like who is the provider but yeah >> yeah go to collab and talk an agent,
936.629–936.639  >> yeah go to collab and talk an agent,
936.639–937.829  >> yeah go to collab and talk an agent, right? And it answers your questions
937.829–937.839  right? And it answers your questions
937.839–939.670  right? And it answers your questions about PyTorch.
939.670–939.680  about PyTorch.
939.680–940.310  about PyTorch. >> Actually,
940.310–940.320  >> Actually,
940.320–941.829  >> Actually, >> I would like to add Go ahead.
941.829–941.839  >> I would like to add Go ahead.
941.839–943.269  >> I would like to add Go ahead. >> Go for it, Andre.
943.269–943.279  >> Go for it, Andre.
943.279–945.590  >> Go for it, Andre. >> So, basically, yes. Um, go ahead and
945.590–945.600  >> So, basically, yes. Um, go ahead and
945.600–947.750  >> So, basically, yes. Um, go ahead and create an issue for us describing
947.750–947.760  create an issue for us describing
947.760–950.790  create an issue for us describing exactly where you want to add the LLM
950.790–950.800  exactly where you want to add the LLM
950.800–953.110  exactly where you want to add the LLM and maybe we can look into it. Yes,
953.110–953.120  and maybe we can look into it. Yes,
953.120–955.030  and maybe we can look into it. Yes, because we have extensive documentation
955.030–955.040  because we have extensive documentation
955.040–958.629  because we have extensive documentation and extensive um uh basically release
958.629–958.639  and extensive um uh basically release
958.639–960.389  and extensive um uh basically release notes and there there are multiple
960.389–960.399  notes and there there are multiple
960.399–962.710  notes and there there are multiple sources of documentation. So I think if
962.710–962.720  sources of documentation. So I think if
962.720–965.269  sources of documentation. So I think if you're more direct about it what you
965.269–965.279  you're more direct about it what you
965.279–967.829  you're more direct about it what you want to do maybe we can come come up
967.829–967.839  want to do maybe we can come come up
967.839–969.749  want to do maybe we can come come up with the solutions for this.
969.749–969.759  with the solutions for this.
969.759–971.350  with the solutions for this. >> Yeah and an issue on GitHub is the right
971.350–971.360  >> Yeah and an issue on GitHub is the right
971.360–972.710  >> Yeah and an issue on GitHub is the right way to do that. Go ahead Joe.
972.710–972.720  way to do that. Go ahead Joe.
972.720–974.949  way to do that. Go ahead Joe. >> Yeah. No, I was just gonna say I mean in
974.949–974.959  >> Yeah. No, I was just gonna say I mean in
974.959–977.110  >> Yeah. No, I was just gonna say I mean in the if the the the search obviously we
977.110–977.120  the if the the the search obviously we
977.120–979.030  the if the the the search obviously we use kind of an LM based search now I
979.030–979.040  use kind of an LM based search now I
979.040–980.870  use kind of an LM based search now I think there and I think you know you can
980.870–980.880  think there and I think you know you can
980.880–983.829  think there and I think you know you can pragmatically um
983.829–983.839  pragmatically um
983.839–986.310  pragmatically um you can pragmatically use like Gemini as
986.310–986.320  you can pragmatically use like Gemini as
986.320–987.670  you can pragmatically use like Gemini as well and I think all of our
987.670–987.680  well and I think all of our
987.680–989.829  well and I think all of our documentation is in distribution right
989.829–989.839  documentation is in distribution right
989.839–992.870  documentation is in distribution right and uh it's it's well well indexed um
992.870–992.880  and uh it's it's well well indexed um
992.880–994.550  and uh it's it's well well indexed um and so you know I go into Google and and
994.550–994.560  and so you know I go into Google and and
994.560–996.389  and so you know I go into Google and and and actually Gemini does a really great
996.389–996.399  and actually Gemini does a really great
996.399–998.150  and actually Gemini does a really great job of like of scouring our
998.150–998.160  job of like of scouring our
998.160–1000.389  job of like of scouring our documentation and and distilling things
1000.389–1000.399  documentation and and distilling things
1000.399–1002.470  documentation and and distilling things down to what I need. So, um, we don't
1002.470–1002.480  down to what I need. So, um, we don't
1002.480–1003.910  down to what I need. So, um, we don't actually even need to do a lot embedded
1003.910–1003.920  actually even need to do a lot embedded
1003.920–1005.189  actually even need to do a lot embedded in our site, but obviously make sure our
1005.189–1005.199  in our site, but obviously make sure our
1005.199–1008.150  in our site, but obviously make sure our docs are updated. Um, and and I mean, we
1008.150–1008.160  docs are updated. Um, and and I mean, we
1008.160–1009.910  docs are updated. Um, and and I mean, we like there's obviously the last year
1009.910–1009.920  like there's obviously the last year
1009.920–1011.829  like there's obviously the last year we've been trending towards how do you
1011.829–1011.839  we've been trending towards how do you
1011.839–1013.590  we've been trending towards how do you build things in a way obviously are good
1013.590–1013.600  build things in a way obviously are good
1013.600–1015.509  build things in a way obviously are good for humans, but also like consumable by
1015.509–1015.519  for humans, but also like consumable by
1015.519–1017.509  for humans, but also like consumable by LMS as well. And so, I think the team
1017.509–1017.519  LMS as well. And so, I think the team
1017.519–1019.189  LMS as well. And so, I think the team has done a great job kind of getting
1019.189–1019.199  has done a great job kind of getting
1019.199–1020.710  has done a great job kind of getting things in a place where machines can
1020.710–1020.720  things in a place where machines can
1020.720–1022.790  things in a place where machines can read things um along with the humans.
1022.790–1022.800  read things um along with the humans.
1022.800–1027.909  read things um along with the humans. So, um, but we continue to improve.
1031.750–1031.760  >> Sounds great. Uh, next question. Um,
1031.760–1035.669  >> Sounds great. Uh, next question. Um, Demila B uh asks, "What is the what is
1035.669–1035.679  Demila B uh asks, "What is the what is
1035.679–1037.270  Demila B uh asks, "What is the what is still the biggest unresolved limitation
1037.270–1037.280  still the biggest unresolved limitation
1037.280–1039.750  still the biggest unresolved limitation in PyTorch today for training large
1039.750–1039.760  in PyTorch today for training large
1039.760–1042.870  in PyTorch today for training large dynamic GNN's especially when the graph
1042.870–1042.880  dynamic GNN's especially when the graph
1042.880–1046.390  dynamic GNN's especially when the graph topology say changes at every step?"
1046.390–1046.400  topology say changes at every step?"
1046.400–1049.750  topology say changes at every step?" That's a great question. Um, who here
1049.750–1049.760  That's a great question. Um, who here
1049.760–1052.310  That's a great question. Um, who here feels most confident talking about graph
1052.310–1052.320  feels most confident talking about graph
1052.320–1055.190  feels most confident talking about graph neural networks?
1055.190–1055.200  neural networks?
1055.200–1056.870  neural networks? Joe, is that an area you've you've
1056.870–1056.880  Joe, is that an area you've you've
1056.880–1059.430  Joe, is that an area you've you've looked into or not so much?
1059.430–1059.440  looked into or not so much?
1059.440–1062.630  looked into or not so much? I'm probably not the GNN expert here. We
1062.630–1062.640  I'm probably not the GNN expert here. We
1062.640–1064.630  I'm probably not the GNN expert here. We have used GNN's in the past for
1064.630–1064.640  have used GNN's in the past for
1064.640–1067.190  have used GNN's in the past for interesting kind of scientific
1067.190–1067.200  interesting kind of scientific
1067.200–1070.310  interesting kind of scientific use cases. Um especially in the
1070.310–1070.320  use cases. Um especially in the
1070.320–1073.590  use cases. Um especially in the chemistry work we did uh with with uh
1073.590–1073.600  chemistry work we did uh with with uh
1073.600–1076.549  chemistry work we did uh with with uh with open chemistry. Um and there's
1076.549–1076.559  with open chemistry. Um and there's
1076.559–1079.590  with open chemistry. Um and there's they're certainly useful um for those
1079.590–1079.600  they're certainly useful um for those
1079.600–1081.190  they're certainly useful um for those types because you have very complex data
1081.190–1081.200  types because you have very complex data
1081.200–1083.190  types because you have very complex data that you're you're trying to embed that
1083.190–1083.200  that you're you're trying to embed that
1083.200–1085.510  that you're you're trying to embed that information into the model. So, a lot of
1085.510–1085.520  information into the model. So, a lot of
1085.520–1087.029  information into the model. So, a lot of different modalities, but I'm I'm
1087.029–1087.039  different modalities, but I'm I'm
1087.039–1089.669  different modalities, but I'm I'm probably far from a GNN expert.
1089.669–1089.679  probably far from a GNN expert.
1089.679–1090.230  probably far from a GNN expert. >> Yeah.
1090.230–1090.240  >> Yeah.
1090.240–1093.270  >> Yeah. >> Yeah. I think the good answer is the one
1093.270–1093.280  >> Yeah. I think the good answer is the one
1093.280–1094.950  >> Yeah. I think the good answer is the one that Andrew made to the previous
1094.950–1094.960  that Andrew made to the previous
1094.960–1096.230  that Andrew made to the previous question like, you know, if you have a
1096.230–1096.240  question like, you know, if you have a
1096.240–1098.150  question like, you know, if you have a particular like I think you should
1098.150–1098.160  particular like I think you should
1098.160–1099.590  particular like I think you should probably file an issue against PyTorch
1099.590–1099.600  probably file an issue against PyTorch
1099.600–1101.750  probably file an issue against PyTorch Geometric and if they say that you know
1101.750–1101.760  Geometric and if they say that you know
1101.760–1103.190  Geometric and if they say that you know that there's a fundamental issue in
1103.190–1103.200  that there's a fundamental issue in
1103.200–1105.029  that there's a fundamental issue in PyTorch, you should file an issue
1105.029–1105.039  PyTorch, you should file an issue
1105.039–1106.630  PyTorch, you should file an issue against us because we don't know until
1106.630–1106.640  against us because we don't know until
1106.640–1109.510  against us because we don't know until we like hear about it. But like I would
1109.510–1109.520  we like hear about it. But like I would
1109.520–1113.270  we like hear about it. But like I would say GNN's is not the most hot topic in
1113.270–1113.280  say GNN's is not the most hot topic in
1113.280–1115.750  say GNN's is not the most hot topic in uh artificial intelligence in the last
1115.750–1115.760  uh artificial intelligence in the last
1115.760–1119.029  uh artificial intelligence in the last 12 months and so we don't like have a
1119.029–1119.039  12 months and so we don't like have a
1119.039–1121.830  12 months and so we don't like have a particular focus on those
1121.830–1121.840  particular focus on those
1121.840–1123.350  particular focus on those but they should work and there shouldn't
1123.350–1123.360  but they should work and there shouldn't
1123.360–1125.270  but they should work and there shouldn't be any limitations other than inherent
1125.270–1125.280  be any limitations other than inherent
1125.280–1126.789  be any limitations other than inherent limitations in the architecture that
1126.789–1126.799  limitations in the architecture that
1126.799–1128.950  limitations in the architecture that like post training is somewhat slow
1128.950–1128.960  like post training is somewhat slow
1128.960–1131.029  like post training is somewhat slow because your graph propagation like
1131.029–1131.039  because your graph propagation like
1131.039–1132.950  because your graph propagation like backward propagation across the graph
1132.950–1132.960  backward propagation across the graph
1132.960–1134.310  backward propagation across the graph like introduce lots and lots of
1134.310–1134.320  like introduce lots and lots of
1134.320–1135.750  like introduce lots and lots of gradients which are somewhat unstable
1135.750–1135.760  gradients which are somewhat unstable
1135.760–1137.750  gradients which are somewhat unstable and you need to like be very careful
1137.750–1137.760  and you need to like be very careful
1137.760–1139.029  and you need to like be very careful because the trading electron rate should
1139.029–1139.039  because the trading electron rate should
1139.039–1143.110  because the trading electron rate should be very low but that's not new. [snorts]
1146.950–1146.960  >> Okay, thank you folks for taking a swing
1146.960–1151.590  >> Okay, thank you folks for taking a swing at a challenging question.
1154.390–1154.400  >> Um Adam's uh question is next. Um how
1154.400–1157.430  >> Um Adam's uh question is next. Um how does an inductor choose between kublos
1157.430–1157.440  does an inductor choose between kublos
1157.440–1159.830  does an inductor choose between kublos cutless and cute DSL kernels and does
1159.830–1159.840  cutless and cute DSL kernels and does
1159.840–1161.830  cutless and cute DSL kernels and does the user have any flexibility here? Is
1161.830–1161.840  the user have any flexibility here? Is
1161.840–1165.110  the user have any flexibility here? Is this cho is this choice exposed um uh
1165.110–1165.120  this cho is this choice exposed um uh
1165.120–1168.390  this cho is this choice exposed um uh for example in FX uh layer?
1168.390–1168.400  for example in FX uh layer?
1168.400–1170.390  for example in FX uh layer? >> So I can probably answer this one. Well
1170.390–1170.400  >> So I can probably answer this one. Well
1170.400–1174.390  >> So I can probably answer this one. Well like inductor tries to do the best perf
1174.390–1174.400  like inductor tries to do the best perf
1174.400–1176.789  like inductor tries to do the best perf right and it like tries to select the
1176.789–1176.799  right and it like tries to select the
1176.799–1179.430  right and it like tries to select the ones that it believes will be better. Uh
1179.430–1179.440  ones that it believes will be better. Uh
1179.440–1181.110  ones that it believes will be better. Uh you can also do max autotune when it
1181.110–1181.120  you can also do max autotune when it
1181.120–1182.870  you can also do max autotune when it should just dutifully iterate over all
1182.870–1182.880  should just dutifully iterate over all
1182.880–1184.470  should just dutifully iterate over all the options and try to pick the one that
1184.470–1184.480  the options and try to pick the one that
1184.480–1185.909  the options and try to pick the one that works for you. Also there is a hardware
1185.909–1185.919  works for you. Also there is a hardware
1185.919–1188.230  works for you. Also there is a hardware limitations and there should be knobs in
1188.230–1188.240  limitations and there should be knobs in
1188.240–1190.150  limitations and there should be knobs in the inductor to say use or doesn't use
1190.150–1190.160  the inductor to say use or doesn't use
1190.160–1193.029  the inductor to say use or doesn't use certain backends but I don't think there
1193.029–1193.039  certain backends but I don't think there
1193.039–1195.909  certain backends but I don't think there is a very easy way to express it through
1195.909–1195.919  is a very easy way to express it through
1195.919–1198.470  is a very easy way to express it through the FX graph but you can suggest this
1198.470–1198.480  the FX graph but you can suggest this
1198.480–1200.150  the FX graph but you can suggest this one as an improvement. So once again
1200.150–1200.160  one as an improvement. So once again
1200.160–1202.310  one as an improvement. So once again file an issue saying that you want this
1202.310–1202.320  file an issue saying that you want this
1202.320–1204.950  file an issue saying that you want this one but I kind of want you to think
1204.950–1204.960  one but I kind of want you to think
1204.960–1207.350  one but I kind of want you to think about it like why do you want to be
1207.350–1207.360  about it like why do you want to be
1207.360–1210.310  about it like why do you want to be explicit like don't we want to converge
1210.310–1210.320  explicit like don't we want to converge
1210.320–1213.350  explicit like don't we want to converge to one model and I suspect that cute DSL
1213.350–1213.360  to one model and I suspect that cute DSL
1213.360–1218.150  to one model and I suspect that cute DSL is probably something we kind of leaning
1218.150–1218.160  is probably something we kind of leaning
1218.160–1219.830  is probably something we kind of leaning towards on the most modern architectures
1219.830–1219.840  towards on the most modern architectures
1219.840–1220.950  towards on the most modern architectures like blackfields and probably have
1220.950–1220.960  like blackfields and probably have
1220.960–1225.669  like blackfields and probably have common verbs but I think like kublast
1225.669–1225.679  common verbs but I think like kublast
1225.679–1226.950  common verbs but I think like kublast will probably give you a much better
1226.950–1226.960  will probably give you a much better
1226.960–1229.909  will probably give you a much better perf on and cutless on an architectures
1229.909–1229.919  perf on and cutless on an architectures
1229.919–1234.870  perf on and cutless on an architectures because there's kind of a cut off there.
1236.950–1236.960  >> Okay, any other uh responses on this
1236.960–1240.390  >> Okay, any other uh responses on this one? Okay, let's move to the next one.
1240.390–1240.400  one? Okay, let's move to the next one.
1240.400–1242.230  one? Okay, let's move to the next one. We have I love that there are so many
1242.230–1242.240  We have I love that there are so many
1242.240–1243.990  We have I love that there are so many audience questions. So, thank you folks
1243.990–1244.000  audience questions. So, thank you folks
1244.000–1246.149  audience questions. So, thank you folks for putting those in. Uh I think that's
1246.149–1246.159  for putting those in. Uh I think that's
1246.159–1249.430  for putting those in. Uh I think that's great. Okay. Um we have somebody from
1249.430–1249.440  great. Okay. Um we have somebody from
1249.440–1251.909  great. Okay. Um we have somebody from LinkedIn who asked uh which PyTorch 20
1251.909–1251.919  LinkedIn who asked uh which PyTorch 20
1251.919–1254.549  LinkedIn who asked uh which PyTorch 20 uh 21.14 uh feature gives the biggest
1254.549–1254.559  uh 21.14 uh feature gives the biggest
1254.559–1256.470  uh 21.14 uh feature gives the biggest performance boost for production LLM
1256.470–1256.480  performance boost for production LLM
1256.480–1259.350  performance boost for production LLM inference and how easy is it uh to
1259.350–1259.360  inference and how easy is it uh to
1259.360–1263.190  inference and how easy is it uh to adopt? Um
1263.190–1263.200  adopt? Um
1263.200–1264.950  adopt? Um I don't know that we did an ablation
1264.950–1264.960  I don't know that we did an ablation
1264.960–1266.630  I don't know that we did an ablation study of the various different features
1266.630–1266.640  study of the various different features
1266.640–1270.549  study of the various different features but does anybody have a um have a
1270.549–1270.559  but does anybody have a um have a
1270.559–1274.549  but does anybody have a um have a thought here? I suspect cudgraphs uh
1274.549–1274.559  thought here? I suspect cudgraphs uh
1274.559–1276.789  thought here? I suspect cudgraphs uh ability to trace more functions, avoid
1276.789–1276.799  ability to trace more functions, avoid
1276.799–1280.149  ability to trace more functions, avoid cudograph breaks and ability to like
1280.149–1280.159  cudograph breaks and ability to like
1280.159–1283.430  cudograph breaks and ability to like trace coms uh would be the one maybe
1283.430–1283.440  trace coms uh would be the one maybe
1283.440–1285.270  trace coms uh would be the one maybe some of the torch compiler features but
1285.270–1285.280  some of the torch compiler features but
1285.280–1286.870  some of the torch compiler features but that depends on the frameworks that
1286.870–1286.880  that depends on the frameworks that
1286.880–1289.669  that depends on the frameworks that you're using for your LLM inference,
1289.669–1289.679  you're using for your LLM inference,
1289.679–1291.270  you're using for your LLM inference, right?
1291.270–1291.280  right?
1291.280–1293.270  right? And they should be pretty easy to adopt.
1293.270–1293.280  And they should be pretty easy to adopt.
1293.280–1294.710  And they should be pretty easy to adopt. So like lots of those features were
1294.710–1294.720  So like lots of those features were
1294.720–1298.390  So like lots of those features were driven by like asks from users such as
1298.390–1298.400  driven by like asks from users such as
1298.400–1302.230  driven by like asks from users such as like VLM or SJAN. So I hope that they
1302.230–1302.240  like VLM or SJAN. So I hope that they
1302.240–1303.750  like VLM or SJAN. So I hope that they will be incorporating those features
1303.750–1303.760  will be incorporating those features
1303.760–1307.110  will be incorporating those features into their frameworks and NVJMS is also
1307.110–1307.120  into their frameworks and NVJMS is also
1307.120–1310.070  into their frameworks and NVJMS is also a great one that probably important for
1310.070–1310.080  a great one that probably important for
1310.080–1313.669  a great one that probably important for performance.
1314.950–1314.960  >> Okay.
1314.960–1318.390  >> Okay. >> Yeah, I think NVJ specifically on black
1318.390–1318.400  >> Yeah, I think NVJ specifically on black
1318.400–1320.470  >> Yeah, I think NVJ specifically on black with low precision I think it's
1320.470–1320.480  with low precision I think it's
1320.480–1325.190  with low precision I think it's important. Yeah.
1329.110–1329.120  >> Okay. Um the next one is from Asam. Uh,
1329.120–1330.630  >> Okay. Um the next one is from Asam. Uh, and I might be mispronouncing there. I
1330.630–1330.640  and I might be mispronouncing there. I
1330.640–1333.110  and I might be mispronouncing there. I apologize if I did. Um, PyTorch, uh,
1333.110–1333.120  apologize if I did. Um, PyTorch, uh,
1333.120–1335.669  apologize if I did. Um, PyTorch, uh, 21.14 makes fault tolerance a first
1335.669–1335.679  21.14 makes fault tolerance a first
1335.679–1338.230  21.14 makes fault tolerance a first class CTN concept. Does the new model
1338.230–1338.240  class CTN concept. Does the new model
1338.240–1340.230  class CTN concept. Does the new model allow a training job to recover from a
1340.230–1340.240  allow a training job to recover from a
1340.240–1341.669  allow a training job to recover from a failed rank without reconstructing the
1341.669–1341.679  failed rank without reconstructing the
1341.679–1343.669  failed rank without reconstructing the entire distributed process group? And
1343.669–1343.679  entire distributed process group? And
1343.679–1345.590  entire distributed process group? And what state is expected to remain valid
1345.590–1345.600  what state is expected to remain valid
1345.600–1349.750  what state is expected to remain valid after recovery?
1352.230–1352.240  Do we have any anyone here who's expert
1352.240–1361.095  Do we have any anyone here who's expert in the new um torchcom's model?
1361.110–1361.120  >> [clears throat]
1361.120–1362.789  >> [clears throat] >> None of us are.
1362.789–1362.799  >> None of us are.
1362.799–1364.070  >> None of us are. >> Okay. Not me.
1364.070–1364.080  >> Okay. Not me.
1364.080–1365.350  >> Okay. Not me. >> I think we're I think we're going to
1365.350–1365.360  >> I think we're I think we're going to
1365.360–1368.789  >> I think we're I think we're going to have to refer this question to um uh to
1368.789–1368.799  have to refer this question to um uh to
1368.799–1370.789  have to refer this question to um uh to the discussion. So this would be a great
1370.789–1370.799  the discussion. So this would be a great
1370.799–1372.870  the discussion. So this would be a great question for example for discuss
1372.870–1372.880  question for example for discuss
1372.880–1376.549  question for example for discuss uh.pytorch or dev discuss. Um so I think
1376.549–1376.559  uh.pytorch or dev discuss. Um so I think
1376.559–1379.270  uh.pytorch or dev discuss. Um so I think uh the the specific uh details about
1379.270–1379.280  uh the the specific uh details about
1379.280–1381.430  uh the the specific uh details about like what part of the state uh would be
1381.430–1381.440  like what part of the state uh would be
1381.440–1383.590  like what part of the state uh would be expected to remain valid uh is the kind
1383.590–1383.600  expected to remain valid uh is the kind
1383.600–1385.909  expected to remain valid uh is the kind of thing one of the distributed uh uh
1385.909–1385.919  of thing one of the distributed uh uh
1385.919–1388.630  of thing one of the distributed uh uh engineers would be really strong
1388.630–1388.640  engineers would be really strong
1388.640–1389.909  engineers would be really strong rice question.
1389.909–1389.919  rice question.
1389.919–1390.390  rice question. >> Yeah.
1390.390–1390.400  >> Yeah.
1390.400–1392.149  >> Yeah. >> Yep. Exactly. But I want to just flag
1392.149–1392.159  >> Yep. Exactly. But I want to just flag
1392.159–1393.830  >> Yep. Exactly. But I want to just flag that this is experimental support and we
1393.830–1393.840  that this is experimental support and we
1393.840–1396.149  that this is experimental support and we indeed like seeking your feedback. So
1396.149–1396.159  indeed like seeking your feedback. So
1396.159–1398.630  indeed like seeking your feedback. So please try to obtain into this like
1398.630–1398.640  please try to obtain into this like
1398.640–1400.710  please try to obtain into this like nickel ch and other CTN default
1400.710–1400.720  nickel ch and other CTN default
1400.720–1402.870  nickel ch and other CTN default tolerance features and let us know if
1402.870–1402.880  tolerance features and let us know if
1402.880–1405.270  tolerance features and let us know if they work for you or not.
1405.270–1405.280  they work for you or not.
1405.280–1408.070  they work for you or not. But yes, like a new model allows for a
1408.070–1408.080  But yes, like a new model allows for a
1408.080–1410.630  But yes, like a new model allows for a more graceful recovery, but it doesn't
1410.630–1410.640  more graceful recovery, but it doesn't
1410.640–1412.310  more graceful recovery, but it doesn't guarantee that it will work out of the
1412.310–1412.320  guarantee that it will work out of the
1412.320–1414.870  guarantee that it will work out of the box. You probably need to like adjust
1414.870–1414.880  box. You probably need to like adjust
1414.880–1418.630  box. You probably need to like adjust your training pipeline to uh like take
1418.630–1418.640  your training pipeline to uh like take
1418.640–1420.390  your training pipeline to uh like take benefit of that and like recognize this
1420.390–1420.400  benefit of that and like recognize this
1420.400–1423.029  benefit of that and like recognize this error as recoverable. But yes, sounds
1423.029–1423.039  error as recoverable. But yes, sounds
1423.039–1427.909  error as recoverable. But yes, sounds like a great discuss question.
1430.070–1430.080  >> Next question.
1430.080–1434.549  >> Next question. Yes, it's the um uh Shiva Shush um uh
1434.549–1434.559  Yes, it's the um uh Shiva Shush um uh
1434.559–1437.830  Yes, it's the um uh Shiva Shush um uh again pronunciation caveat on Apple
1437.830–1437.840  again pronunciation caveat on Apple
1437.840–1440.390  again pronunciation caveat on Apple silicon uh 21.14 adds uh native linear
1440.390–1440.400  silicon uh 21.14 adds uh native linear
1440.400–1442.630  silicon uh 21.14 adds uh native linear algebra and more metal kernels uh for
1442.630–1442.640  algebra and more metal kernels uh for
1442.640–1445.029  algebra and more metal kernels uh for LLM decode on unified memory. Which path
1445.029–1445.039  LLM decode on unified memory. Which path
1445.039–1447.669  LLM decode on unified memory. Which path actually moves the needle uh versus 2.13
1447.669–1447.679  actually moves the needle uh versus 2.13
1447.679–1449.830  actually moves the needle uh versus 2.13 the new metal kernels or inductor
1449.830–1449.840  the new metal kernels or inductor
1449.840–1452.549  the new metal kernels or inductor picking up better epilogs? Any measured
1452.549–1452.559  picking up better epilogs? Any measured
1452.559–1455.110  picking up better epilogs? Any measured tokens per second deltas um on the M
1455.110–1455.120  tokens per second deltas um on the M
1455.120–1457.990  tokens per second deltas um on the M series or fixed to code loop? And I I
1457.990–1458.000  series or fixed to code loop? And I I
1458.000–1459.269  series or fixed to code loop? And I I believe Nikita, you've been really
1459.269–1459.279  believe Nikita, you've been really
1459.279–1460.710  believe Nikita, you've been really active in the past with some of the
1460.710–1460.720  active in the past with some of the
1460.720–1462.789  active in the past with some of the Apple changes. So I'm gonna toss this
1462.789–1462.799  Apple changes. So I'm gonna toss this
1462.799–1464.230  Apple changes. So I'm gonna toss this one to you.
1464.230–1464.240  one to you.
1464.240–1465.750  one to you. >> Sure. Well, first of all, we don't
1465.750–1465.760  >> Sure. Well, first of all, we don't
1465.760–1467.990  >> Sure. Well, first of all, we don't specifically measure and also it's very
1467.990–1468.000  specifically measure and also it's very
1468.000–1470.470  specifically measure and also it's very subjective because like different M
1470.470–1470.480  subjective because like different M
1470.480–1472.310  subjective because like different M series have a different behavior. But
1472.310–1472.320  series have a different behavior. But
1472.320–1474.149  series have a different behavior. But there is one feature that was introduced
1474.149–1474.159  there is one feature that was introduced
1474.159–1478.149  there is one feature that was introduced in 214. we use uh like a new apple
1478.149–1478.159  in 214. we use uh like a new apple
1478.159–1480.310  in 214. we use uh like a new apple silicon or like metal framework feature
1480.310–1480.320  silicon or like metal framework feature
1480.320–1481.750  silicon or like metal framework feature called metal performance primitives
1481.750–1481.760  called metal performance primitives
1481.760–1485.029  called metal performance primitives which is similar to uh I don't know
1485.029–1485.039  which is similar to uh I don't know
1485.039–1487.669  which is similar to uh I don't know probably to cut in a sense so it's very
1487.669–1487.679  probably to cut in a sense so it's very
1487.679–1489.669  probably to cut in a sense so it's very different it allows one to express
1489.669–1489.679  different it allows one to express
1489.679–1491.909  different it allows one to express matrix multiplication it's done in eager
1491.909–1491.919  matrix multiplication it's done in eager
1491.919–1495.029  matrix multiplication it's done in eager mode and it can in some cases produce
1495.029–1495.039  mode and it can in some cases produce
1495.039–1497.029  mode and it can in some cases produce just of this particular operation like
1497.029–1497.039  just of this particular operation like
1497.039–1499.750  just of this particular operation like 3x or even sometimes measured 5x but
1499.750–1499.760  3x or even sometimes measured 5x but
1499.760–1501.350  3x or even sometimes measured 5x but this is more like a synthetic benchmark
1501.350–1501.360  this is more like a synthetic benchmark
1501.360–1503.430  this is more like a synthetic benchmark not an end to end one performance boost
1503.430–1503.440  not an end to end one performance boost
1503.440–1506.149  not an end to end one performance boost so you should give it tries there is no
1506.149–1506.159  so you should give it tries there is no
1506.159–1508.149  so you should give it tries there is no inductor peing better epilogues
1508.149–1508.159  inductor peing better epilogues
1508.159–1511.430  inductor peing better epilogues necessarily for Apple silicon and yes we
1511.430–1511.440  necessarily for Apple silicon and yes we
1511.440–1512.950  necessarily for Apple silicon and yes we we don't do measurements because again
1512.950–1512.960  we don't do measurements because again
1512.960–1514.630  we don't do measurements because again it's kind of you cannot measure on the
1514.630–1514.640  it's kind of you cannot measure on the
1514.640–1516.390  it's kind of you cannot measure on the pietorch alone you need to measure on a
1516.390–1516.400  pietorch alone you need to measure on a
1516.400–1518.470  pietorch alone you need to measure on a particular framework but you can
1518.470–1518.480  particular framework but you can
1518.480–1521.750  particular framework but you can probably check those numbers at hugging
1521.750–1521.760  probably check those numbers at hugging
1521.760–1523.990  probably check those numbers at hugging face transformers on one of their um
1523.990–1524.000  face transformers on one of their um
1524.000–1525.590  face transformers on one of their um frameworks if this is what you're using
1525.590–1525.600  frameworks if this is what you're using
1525.600–1532.070  frameworks if this is what you're using for your inference pipelines
1537.350–1537.360  >> okay Um yes, let's do so. Uh Shibash had
1537.360–1538.950  >> okay Um yes, let's do so. Uh Shibash had three several questions in a row, so I
1538.950–1538.960  three several questions in a row, so I
1538.960–1539.909  three several questions in a row, so I don't know if we'll get to all of them,
1539.909–1539.919  don't know if we'll get to all of them,
1539.919–1541.222  don't know if we'll get to all of them, but let's take this next one here.
1541.222–1541.232  but let's take this next one here.
1541.232–1543.190  but let's take this next one here. [snorts] Uh the declarative uh dynamic
1543.190–1543.200  [snorts] Uh the declarative uh dynamic
1543.200–1545.029  [snorts] Uh the declarative uh dynamic spec looks useful for Asian workloads
1545.029–1545.039  spec looks useful for Asian workloads
1545.039–1547.510  spec looks useful for Asian workloads where context length um changes
1547.510–1547.520  where context length um changes
1547.520–1550.070  where context length um changes midsession. Uh if the live shape set
1550.070–1550.080  midsession. Uh if the live shape set
1550.080–1552.870  midsession. Uh if the live shape set expands beyond the declared spec, does
1552.870–1552.880  expands beyond the declared spec, does
1552.880–1554.870  expands beyond the declared spec, does inductor recompile quietly or can you
1554.870–1554.880  inductor recompile quietly or can you
1554.880–1557.350  inductor recompile quietly or can you make it a hard fail uh so that you can
1557.350–1557.360  make it a hard fail uh so that you can
1557.360–1561.029  make it a hard fail uh so that you can have more predictable latency? Um does
1561.029–1561.039  have more predictable latency? Um does
1561.039–1564.870  have more predictable latency? Um does anybody know the answer to this one? Um
1564.870–1564.880  anybody know the answer to this one? Um
1564.880–1567.190  anybody know the answer to this one? Um >> I don't know on top of my head but
1567.190–1567.200  >> I don't know on top of my head but
1567.200–1569.110  >> I don't know on top of my head but knowing the development patterns of the
1569.110–1569.120  knowing the development patterns of the
1569.120–1571.750  knowing the development patterns of the inductor I suspect there should be an op
1571.750–1571.760  inductor I suspect there should be an op
1571.760–1574.390  inductor I suspect there should be an op to make it fail
1574.390–1574.400  to make it fail
1574.400–1576.870  to make it fail >> but there also should be a knob like
1576.870–1576.880  >> but there also should be a knob like
1576.880–1579.830  >> but there also should be a knob like that was a default orch compile behavior
1579.830–1579.840  that was a default orch compile behavior
1579.840–1581.430  that was a default orch compile behavior for quite some time that like if it
1581.430–1581.440  for quite some time that like if it
1581.440–1582.950  for quite some time that like if it fails to do something instead of ering
1582.950–1582.960  fails to do something instead of ering
1582.960–1585.190  fails to do something instead of ering out it just falls back to either or like
1585.190–1585.200  out it just falls back to either or like
1585.200–1587.029  out it just falls back to either or like to the previous behavior. So there are
1587.029–1587.039  to the previous behavior. So there are
1587.039–1588.870  to the previous behavior. So there are both nodes exist but you should ask this
1588.870–1588.880  both nodes exist but you should ask this
1588.880–1591.350  both nodes exist but you should ask this question and de discuss. Um it's kind of
1591.350–1591.360  question and de discuss. Um it's kind of
1591.360–1594.230  question and de discuss. Um it's kind of yeah hard to know all the specifics.
1594.230–1594.240  yeah hard to know all the specifics.
1594.240–1597.029  yeah hard to know all the specifics. >> Yeah I feel I I feel I feel like the if
1597.029–1597.039  >> Yeah I feel I I feel I feel like the if
1597.039–1598.789  >> Yeah I feel I I feel I feel like the if we don't already have a knob to do this
1598.789–1598.799  we don't already have a knob to do this
1598.799–1600.390  we don't already have a knob to do this this would be a knob that that the team
1600.390–1600.400  this would be a knob that that the team
1600.400–1603.029  this would be a knob that that the team would likely receive. Well um because
1603.029–1603.039  would likely receive. Well um because
1603.039–1605.510  would likely receive. Well um because that that latency uh predictability is
1605.510–1605.520  that that latency uh predictability is
1605.520–1606.870  that that latency uh predictability is is pretty important to a lot of use
1606.870–1606.880  is pretty important to a lot of use
1606.880–1609.909  is pretty important to a lot of use cases. Okay. Um let's go ahead and move
1609.909–1609.919  cases. Okay. Um let's go ahead and move
1609.919–1613.029  cases. Okay. Um let's go ahead and move on to the to the next uh question in the
1613.029–1613.039  on to the to the next uh question in the
1613.039–1616.950  on to the to the next uh question in the list here. Um, how will dropping a CUDA
1616.950–1616.960  list here. Um, how will dropping a CUDA
1616.960–1620.470  list here. Um, how will dropping a CUDA 12 wheels in 2.15 affect downstream uh
1620.470–1620.480  12 wheels in 2.15 affect downstream uh
1620.480–1624.310  12 wheels in 2.15 affect downstream uh C++ libr uh libraries uh or binaries for
1624.310–1624.320  C++ libr uh libraries uh or binaries for
1624.320–1626.549  C++ libr uh libraries uh or binaries for for projects such as to sharp, but I
1626.549–1626.559  for projects such as to sharp, but I
1626.559–1629.190  for projects such as to sharp, but I know there's there's others. Um, Andre,
1629.190–1629.200  know there's there's others. Um, Andre,
1629.200–1630.390  know there's there's others. Um, Andre, do you want to take this one?
1630.390–1630.400  do you want to take this one?
1630.400–1632.470  do you want to take this one? >> Yeah. Yes. Uh, so basically I think
1632.470–1632.480  >> Yeah. Yes. Uh, so basically I think
1632.480–1635.110  >> Yeah. Yes. Uh, so basically I think right now it would be good time for you
1635.110–1635.120  right now it would be good time for you
1635.120–1637.590  right now it would be good time for you to start planning to migrating for coder
1637.590–1637.600  to start planning to migrating for coder
1637.600–1640.549  to start planning to migrating for coder 13 and probably the best version would
1640.549–1640.559  13 and probably the best version would
1640.559–1643.590  13 and probably the best version would be 13.2 two since this is going to be
1643.590–1643.600  be 13.2 two since this is going to be
1643.600–1647.350  be 13.2 two since this is going to be our um stable version for the next 2.15
1647.350–1647.360  our um stable version for the next 2.15
1647.360–1650.950  our um stable version for the next 2.15 release. Um and yeah, I think it's and
1650.950–1650.960  release. Um and yeah, I think it's and
1650.960–1654.390  release. Um and yeah, I think it's and if you still want to use CUDA2.x, you
1654.390–1654.400  if you still want to use CUDA2.x, you
1654.400–1656.950  if you still want to use CUDA2.x, you should be able to compile yourself from
1656.950–1656.960  should be able to compile yourself from
1656.960–1659.350  should be able to compile yourself from the source. So this is should still be
1659.350–1659.360  the source. So this is should still be
1659.360–1662.149  the source. So this is should still be supported. So you can take a look at how
1662.149–1662.159  supported. So you can take a look at how
1662.159–1664.630  supported. So you can take a look at how we compile liptor binaries and basically
1664.630–1664.640  we compile liptor binaries and basically
1664.640–1666.950  we compile liptor binaries and basically duplicate the workflow and compile it
1666.950–1666.960  duplicate the workflow and compile it
1666.960–1670.310  duplicate the workflow and compile it yourself on your workers. um that that
1670.310–1670.320  yourself on your workers. um that that
1670.320–1672.710  yourself on your workers. um that that could be an option for short term like
1672.710–1672.720  could be an option for short term like
1672.720–1676.470  could be an option for short term like if you cannot migrate uh faster um I
1676.470–1676.480  if you cannot migrate uh faster um I
1676.480–1678.070  if you cannot migrate uh faster um I don't know maybe Nikita wants to chime
1678.070–1678.080  don't know maybe Nikita wants to chime
1678.080–1679.909  don't know maybe Nikita wants to chime in here any additional
1679.909–1679.919  in here any additional
1679.919–1685.190  in here any additional >> um sure I guess like uh uh two
1685.190–1685.200  >> um sure I guess like uh uh two
1685.200–1687.669  >> um sure I guess like uh uh two question like two two points one is that
1687.669–1687.679  question like two two points one is that
1687.679–1690.149  question like two two points one is that uh like if you're on an older GPU
1690.149–1690.159  uh like if you're on an older GPU
1690.159–1691.909  uh like if you're on an older GPU unfortunately CUDA 13 doesn't support
1691.909–1691.919  unfortunately CUDA 13 doesn't support
1691.919–1694.389  unfortunately CUDA 13 doesn't support some of them so you have to stay on the
1694.389–1694.399  some of them so you have to stay on the
1694.399–1696.470  some of them so you have to stay on the older PyTorch versions but also one
1696.470–1696.480  older PyTorch versions but also one
1696.480–1698.710  older PyTorch versions but also one should not expect any performance
1698.710–1698.720  should not expect any performance
1698.720–1703.029  should not expect any performance uh gains or uh like new features for
1703.029–1703.039  uh gains or uh like new features for
1703.039–1704.870  uh gains or uh like new features for this older hardware with new releases.
1704.870–1704.880  this older hardware with new releases.
1704.880–1706.630  this older hardware with new releases. So like there is no reason for you to
1706.630–1706.640  So like there is no reason for you to
1706.640–1709.750  So like there is no reason for you to upgrade. You can continue using 214 uh
1709.750–1709.760  upgrade. You can continue using 214 uh
1709.760–1713.430  upgrade. You can continue using 214 uh for your older like pipelines
1713.430–1713.440  for your older like pipelines
1713.440–1715.350  for your older like pipelines and TPU binaries will still be there,
1715.350–1715.360  and TPU binaries will still be there,
1715.360–1717.590  and TPU binaries will still be there, right? I don't know how much to sharp is
1717.590–1717.600  right? I don't know how much to sharp is
1717.600–1719.750  right? I don't know how much to sharp is indeed about like I guess it's most
1719.750–1719.760  indeed about like I guess it's most
1719.760–1721.909  indeed about like I guess it's most frequently used on Windows and I don't
1721.909–1721.919  frequently used on Windows and I don't
1721.919–1726.630  frequently used on Windows and I don't know how much of that is like
1726.630–1726.640  know how much of that is like
1726.640–1729.110  know how much of that is like relevant to people on Windows like using
1729.110–1729.120  relevant to people on Windows like using
1729.120–1730.950  relevant to people on Windows like using blackville features like like VFP4
1730.950–1730.960  blackville features like like VFP4
1730.960–1735.110  blackville features like like VFP4 because like not available. So use 214
1735.110–1735.120  because like not available. So use 214
1735.120–1739.350  because like not available. So use 214 but also yeah like CUDA 12 was released
1739.350–1739.360  but also yeah like CUDA 12 was released
1739.360–1741.750  but also yeah like CUDA 12 was released five years ago if I'm not mistaken. So
1741.750–1741.760  five years ago if I'm not mistaken. So
1741.760–1743.590  five years ago if I'm not mistaken. So it's it's a good time to upgrade like
1743.590–1743.600  it's it's a good time to upgrade like
1743.600–1745.110  it's it's a good time to upgrade like ask yourself why do you need to be on
1745.110–1745.120  ask yourself why do you need to be on
1745.120–1748.549  ask yourself why do you need to be on CUDA 12 in 2026 or like heading towards
1748.549–1748.559  CUDA 12 in 2026 or like heading towards
1748.559–1751.480  CUDA 12 in 2026 or like heading towards 2027
1751.480–1751.490  2027
1751.490–1751.990  2027 [snorts]
1751.990–1752.000  [snorts]
1752.000–1753.510  [snorts] >> though I' though I've known people who
1753.510–1753.520  >> though I' though I've known people who
1753.520–1755.110  >> though I' though I've known people who have stayed on old CUDA versions for a
1755.110–1755.120  have stayed on old CUDA versions for a
1755.120–1758.630  have stayed on old CUDA versions for a long time but but yes
1758.630–1758.640  long time but but yes
1758.640–1760.310  long time but but yes okay
1760.310–1760.320  okay
1760.320–1763.269  okay next question
1763.269–1763.279  next question
1763.279–1765.110  next question um okay so this one is a simple one and
1765.110–1765.120  um okay so this one is a simple one and
1765.120–1768.070  um okay so this one is a simple one and this uh came from the the community uh
1768.070–1768.080  this uh came from the the community uh
1768.080–1770.070  this uh came from the the community uh questions ahead of time uh if you're on
1770.070–1770.080  questions ahead of time uh if you're on
1770.080–1775.190  questions ahead of time uh if you're on AMD what change for uh rockcomm in 2.14.
1775.190–1775.200  AMD what change for uh rockcomm in 2.14.
1775.200–1777.190  AMD what change for uh rockcomm in 2.14. Andre, do you want to talk to this one?
1777.190–1777.200  Andre, do you want to talk to this one?
1777.200–1779.990  Andre, do you want to talk to this one? >> Yes. Yes. So, basically for romcom
1779.990–1780.000  >> Yes. Yes. So, basically for romcom
1780.000–1783.190  >> Yes. Yes. So, basically for romcom specifically, uh we removed the rocom
1783.190–1783.200  specifically, uh we removed the rocom
1783.200–1787.110  specifically, uh we removed the rocom 7.1 support. Uh no wheels are no longer
1787.110–1787.120  7.1 support. Uh no wheels are no longer
1787.120–1792.549  7.1 support. Uh no wheels are no longer shipped with 214 PyTorch. So, um uh if
1792.549–1792.559  shipped with 214 PyTorch. So, um uh if
1792.559–1795.190  shipped with 214 PyTorch. So, um uh if you're on roam 7.1, I think you need to
1795.190–1795.200  you're on roam 7.1, I think you need to
1795.200–1799.990  you're on roam 7.1, I think you need to migrate to either Rocomm 7.2 or 714. Uh
1799.990–1800.000  migrate to either Rocomm 7.2 or 714. Uh
1800.000–1802.310  migrate to either Rocomm 7.2 or 714. Uh those are two versions that are
1802.310–1802.320  those are two versions that are
1802.320–1806.630  those are two versions that are supported currently by 214 PyTorch. Um
1806.630–1806.640  supported currently by 214 PyTorch. Um
1806.640–1810.230  supported currently by 214 PyTorch. Um 714 is the latest one and it is using uh
1810.230–1810.240  714 is the latest one and it is using uh
1810.240–1814.710  714 is the latest one and it is using uh the rock pdk 7.2 it's the same version
1814.710–1814.720  the rock pdk 7.2 it's the same version
1814.720–1816.789  the rock pdk 7.2 it's the same version that we also shipped with previous
1816.789–1816.799  that we also shipped with previous
1816.799–1819.669  that we also shipped with previous release of torch. So it's uh more
1819.669–1819.679  release of torch. So it's uh more
1819.679–1822.789  release of torch. So it's uh more similar to what rocom 7.1 was. So I
1822.789–1822.799  similar to what rocom 7.1 was. So I
1822.799–1824.950  similar to what rocom 7.1 was. So I think you so you have a choice
1824.950–1824.960  think you so you have a choice
1824.960–1826.870  think you so you have a choice basically. I think the best way is to
1826.870–1826.880  basically. I think the best way is to
1826.880–1831.190  basically. I think the best way is to migrate straight to 714. Um but yeah, I
1831.190–1831.200  migrate straight to 714. Um but yeah, I
1831.200–1835.321  migrate straight to 714. Um but yeah, I think that's about it. Yes.
1835.321–1835.331  think that's about it. Yes.
1835.331–1837.510  think that's about it. Yes. [snorts]
1837.830–1837.840  >> Okay.
1837.840–1839.590  >> Okay. >> And there are some advanced features
1839.590–1839.600  >> And there are some advanced features
1839.600–1842.549  >> And there are some advanced features which are available in 714 like he file
1842.549–1842.559  which are available in 714 like he file
1842.559–1844.870  which are available in 714 like he file which is similar to coup file like
1844.870–1844.880  which is similar to coup file like
1844.880–1847.990  which is similar to coup file like direct storage access from GPUs and you
1847.990–1848.000  direct storage access from GPUs and you
1848.000–1850.470  direct storage access from GPUs and you can use more hip solver functions for
1850.470–1850.480  can use more hip solver functions for
1850.480–1854.630  can use more hip solver functions for example torch and alf uses that one.
1854.630–1854.640  example torch and alf uses that one.
1854.640–1856.950  example torch and alf uses that one. >> Yes. And the packaging is actually a lot
1856.950–1856.960  >> Yes. And the packaging is actually a lot
1856.960–1863.269  >> Yes. And the packaging is actually a lot better and a lot more modern for 714.
1865.029–1865.039  >> Okay.
1865.039–1867.830  >> Okay. Um I think our next question is going to
1867.830–1867.840  Um I think our next question is going to
1867.840–1869.750  Um I think our next question is going to be kind of a little bit of a broader
1869.750–1869.760  be kind of a little bit of a broader
1869.760–1872.950  be kind of a little bit of a broader question. Um uh so it seems like the
1872.950–1872.960  question. Um uh so it seems like the
1872.960–1875.830  question. Um uh so it seems like the diversity in the Okay, here we are. Uh
1875.830–1875.840  diversity in the Okay, here we are. Uh
1875.840–1877.190  diversity in the Okay, here we are. Uh it seems like the diversity in hardware
1877.190–1877.200  it seems like the diversity in hardware
1877.200–1879.590  it seems like the diversity in hardware in the AI world is exploding lately. At
1879.590–1879.600  in the AI world is exploding lately. At
1879.600–1882.470  in the AI world is exploding lately. At least that's what I see on on on uh uh
1882.470–1882.480  least that's what I see on on on uh uh
1882.480–1886.470  least that's what I see on on on uh uh in my news passing by. uh my uh screen
1886.470–1886.480  in my news passing by. uh my uh screen
1886.480–1888.549  in my news passing by. uh my uh screen every day. Uh how do we think about
1888.549–1888.559  every day. Uh how do we think about
1888.559–1891.269  every day. Uh how do we think about platform support um within the PyTorch
1891.269–1891.279  platform support um within the PyTorch
1891.279–1893.350  platform support um within the PyTorch community? I'd love to hear from from
1893.350–1893.360  community? I'd love to hear from from
1893.360–1895.590  community? I'd love to hear from from Joe on this and then I'm sure Andrea and
1895.590–1895.600  Joe on this and then I'm sure Andrea and
1895.600–1897.190  Joe on this and then I'm sure Andrea and Nikita may also want to chime in with
1897.190–1897.200  Nikita may also want to chime in with
1897.200–1898.149  Nikita may also want to chime in with details.
1898.149–1898.159  details.
1898.159–1899.669  details. >> Yeah,
1899.669–1899.679  >> Yeah,
1899.679–1902.549  >> Yeah, >> I mean we know we've been I'm coming at
1902.549–1902.559  >> I mean we know we've been I'm coming at
1902.559–1905.430  >> I mean we know we've been I'm coming at Harbor 180 for a long time with PyTorch.
1905.430–1905.440  Harbor 180 for a long time with PyTorch.
1905.440–1907.190  Harbor 180 for a long time with PyTorch. I think we've been trying uh to support
1907.190–1907.200  I think we've been trying uh to support
1907.200–1909.110  I think we've been trying uh to support a lot of different backends. We there's
1909.110–1909.120  a lot of different backends. We there's
1909.120–1911.750  a lot of different backends. We there's working groups in the in the foundation.
1911.750–1911.760  working groups in the in the foundation.
1911.760–1915.750  working groups in the in the foundation. I think you know in some ways like you
1915.750–1915.760  I think you know in some ways like you
1915.760–1917.110  I think you know in some ways like you know there's a couple different angles.
1917.110–1917.120  know there's a couple different angles.
1917.120–1918.549  know there's a couple different angles. One obviously we wanted to have a
1918.549–1918.559  One obviously we wanted to have a
1918.559–1921.190  One obviously we wanted to have a diverse set of of hardware vendors and
1921.190–1921.200  diverse set of of hardware vendors and
1921.200–1922.870  diverse set of of hardware vendors and and a hardware ecosystem to give choice
1922.870–1922.880  and a hardware ecosystem to give choice
1922.880–1925.269  and a hardware ecosystem to give choice to developers because not just at the
1925.269–1925.279  to developers because not just at the
1925.279–1926.389  to developers because not just at the cloud. I think everyone sort of thinks
1926.389–1926.399  cloud. I think everyone sort of thinks
1926.399–1927.909  cloud. I think everyone sort of thinks of like the cloud side is like the
1927.909–1927.919  of like the cloud side is like the
1927.919–1929.909  of like the cloud side is like the default but like running on device
1929.909–1929.919  default but like running on device
1929.919–1931.909  default but like running on device running you know um within wearables
1931.909–1931.919  running you know um within wearables
1931.919–1934.310  running you know um within wearables like there's there's like a lot going on
1934.310–1934.320  like there's there's like a lot going on
1934.320–1935.909  like there's there's like a lot going on in different hardware architectures and
1935.909–1935.919  in different hardware architectures and
1935.919–1938.070  in different hardware architectures and that's why this like compiler and kernel
1938.070–1938.080  that's why this like compiler and kernel
1938.080–1940.470  that's why this like compiler and kernel ecosystems is so diverse. I think that's
1940.470–1940.480  ecosystems is so diverse. I think that's
1940.480–1942.549  ecosystems is so diverse. I think that's like one piece. I think pragmatically
1942.549–1942.559  like one piece. I think pragmatically
1942.559–1945.029  like one piece. I think pragmatically though like like compute is scarce these
1945.029–1945.039  though like like compute is scarce these
1945.039–1947.750  though like like compute is scarce these days and you know being able to run on
1947.750–1947.760  days and you know being able to run on
1947.760–1949.430  days and you know being able to run on Nvidia, being able to run on AMD, being
1949.430–1949.440  Nvidia, being able to run on AMD, being
1949.440–1950.950  Nvidia, being able to run on AMD, being able to run on TPUs. I think it's
1950.950–1950.960  able to run on TPUs. I think it's
1950.960–1954.789  able to run on TPUs. I think it's actually um you know things have like
1954.789–1954.799  actually um you know things have like
1954.799–1956.710  actually um you know things have like you know big labs or folks that that
1956.710–1956.720  you know big labs or folks that that
1956.720–1958.950  you know big labs or folks that that need a lot of compute have have had to
1958.950–1958.960  need a lot of compute have have had to
1958.960–1961.350  need a lot of compute have have had to learn how to run on on you know a lot of
1961.350–1961.360  learn how to run on on you know a lot of
1961.360–1963.190  learn how to run on on you know a lot of different architectures. I think like if
1963.190–1963.200  different architectures. I think like if
1963.200–1964.950  different architectures. I think like if I look at anthropic they run on Nvidia,
1964.950–1964.960  I look at anthropic they run on Nvidia,
1964.960–1967.669  I look at anthropic they run on Nvidia, they run on Tranium, they run on TPUs.
1967.669–1967.679  they run on Tranium, they run on TPUs.
1967.679–1970.630  they run on Tranium, they run on TPUs. Um even in like different workflows like
1970.630–1970.640  Um even in like different workflows like
1970.640–1971.909  Um even in like different workflows like I know we're talking a lot about
1971.909–1971.919  I know we're talking a lot about
1971.919–1974.950  I know we're talking a lot about training typically but you know training
1974.950–1974.960  training typically but you know training
1974.960–1976.710  training typically but you know training RL like we're talking about running on
1976.710–1976.720  RL like we're talking about running on
1976.720–1979.269  RL like we're talking about running on GPUs. We run environments in CPUs. So
1979.269–1979.279  GPUs. We run environments in CPUs. So
1979.279–1980.789  GPUs. We run environments in CPUs. So all of a sudden you see the Intel stock
1980.789–1980.799  all of a sudden you see the Intel stock
1980.799–1982.389  all of a sudden you see the Intel stock up these days because they're selling a
1982.389–1982.399  up these days because they're selling a
1982.399–1985.590  up these days because they're selling a lot more CPUs for RL which is good. Um
1985.590–1985.600  lot more CPUs for RL which is good. Um
1985.600–1988.149  lot more CPUs for RL which is good. Um you know when you you run reinforcement
1988.149–1988.159  you know when you you run reinforcement
1988.159–1989.110  you know when you you run reinforcement learning you're running a lot of
1989.110–1989.120  learning you're running a lot of
1989.120–1991.350  learning you're running a lot of inference. um at inference time you're
1991.350–1991.360  inference. um at inference time you're
1991.360–1993.029  inference. um at inference time you're running a ton of inference obviously
1993.029–1993.039  running a ton of inference obviously
1993.039–1995.190  running a ton of inference obviously like you're doing a lot of compute time
1995.190–1995.200  like you're doing a lot of compute time
1995.200–1997.110  like you're doing a lot of compute time inference time compute um scaling that
1997.110–1997.120  inference time compute um scaling that
1997.120–1999.430  inference time compute um scaling that up in different axes um and so I think
1999.430–1999.440  up in different axes um and so I think
1999.440–2001.669  up in different axes um and so I think like you know you might run that on on
2001.669–2001.679  like you know you might run that on on
2001.679–2003.350  like you know you might run that on on different architecture you might run AMD
2003.350–2003.360  different architecture you might run AMD
2003.360–2004.789  different architecture you might run AMD you might run on on on different things
2004.789–2004.799  you might run on on on different things
2004.799–2006.310  you might run on on on different things so I think the hardware diversity is
2006.310–2006.320  so I think the hardware diversity is
2006.320–2008.230  so I think the hardware diversity is like is great for developers and I think
2008.230–2008.240  like is great for developers and I think
2008.240–2010.549  like is great for developers and I think that even going back like a year or
2010.549–2010.559  that even going back like a year or
2010.559–2012.549  that even going back like a year or longer we've been talking about this and
2012.549–2012.559  longer we've been talking about this and
2012.559–2014.470  longer we've been talking about this and um for PyTorch even back in we've been
2014.470–2014.480  um for PyTorch even back in we've been
2014.480–2017.590  um for PyTorch even back in we've been working with AMD since what 2018 um I'm
2017.590–2017.600  working with AMD since what 2018 um I'm
2017.600–2019.750  working with AMD since what 2018 um I'm so you remember those days right So, I I
2019.750–2019.760  so you remember those days right So, I I
2019.760–2021.110  so you remember those days right So, I I think it's it's I I wouldn't even say
2021.110–2021.120  think it's it's I I wouldn't even say
2021.120–2022.870  think it's it's I I wouldn't even say it's exploding lately. I actually think
2022.870–2022.880  it's exploding lately. I actually think
2022.880–2024.710  it's exploding lately. I actually think it's been exploding for a long time and
2024.710–2024.720  it's been exploding for a long time and
2024.720–2027.750  it's been exploding for a long time and I think um we just continue to grow and
2027.750–2027.760  I think um we just continue to grow and
2027.760–2029.029  I think um we just continue to grow and obviously Nvidia gets a lot of the
2029.029–2029.039  obviously Nvidia gets a lot of the
2029.039–2031.509  obviously Nvidia gets a lot of the headlines as they as they well deserve.
2031.509–2031.519  headlines as they as they well deserve.
2031.519–2032.950  headlines as they as they well deserve. Um but there's definitely a lot of
2032.950–2032.960  Um but there's definitely a lot of
2032.960–2034.389  Um but there's definitely a lot of architectures out there that are are
2034.389–2034.399  architectures out there that are are
2034.399–2037.909  architectures out there that are are needed and um and it's just going to
2037.909–2037.919  needed and um and it's just going to
2037.919–2039.350  needed and um and it's just going to continue to kind of grow in different
2039.350–2039.360  continue to kind of grow in different
2039.360–2041.669  continue to kind of grow in different directions. Um, and I think we're going
2041.669–2041.679  directions. Um, and I think we're going
2041.679–2043.590  directions. Um, and I think we're going to get inference is obviously much more
2043.590–2043.600  to get inference is obviously much more
2043.600–2046.070  to get inference is obviously much more important these days for many. Um, and
2046.070–2046.080  important these days for many. Um, and
2046.080–2047.830  important these days for many. Um, and it has to be much more efficient to be
2047.830–2047.840  it has to be much more efficient to be
2047.840–2049.430  it has to be much more efficient to be able to scale and that's just going to
2049.430–2049.440  able to scale and that's just going to
2049.440–2050.550  able to scale and that's just going to necessitate different types of
2050.550–2050.560  necessitate different types of
2050.560–2052.149  necessitate different types of architectures. Disagregated inference
2052.149–2052.159  architectures. Disagregated inference
2052.159–2053.430  architectures. Disagregated inference for example where you separate pre-fill
2053.430–2053.440  for example where you separate pre-fill
2053.440–2055.909  for example where you separate pre-fill and decode um is a major trend right now
2055.909–2055.919  and decode um is a major trend right now
2055.919–2056.950  and decode um is a major trend right now and has been for the last several
2056.950–2056.960  and has been for the last several
2056.960–2059.270  and has been for the last several months. Um, so I think that's it's going
2059.270–2059.280  months. Um, so I think that's it's going
2059.280–2060.950  months. Um, so I think that's it's going to continue to innovate obviously where
2060.950–2060.960  to continue to innovate obviously where
2060.960–2062.790  to continue to innovate obviously where the workloads are going to be. So would
2062.790–2062.800  the workloads are going to be. So would
2062.800–2065.909  the workloads are going to be. So would love others to to chime in though.
2065.909–2065.919  love others to to chime in though.
2065.919–2070.230  love others to to chime in though. >> Yeah. Um, Nikita and Andre, do you want
2070.230–2070.240  >> Yeah. Um, Nikita and Andre, do you want
2070.240–2071.990  >> Yeah. Um, Nikita and Andre, do you want to say anything about distributed or
2071.990–2072.000  to say anything about distributed or
2072.000–2074.710  to say anything about distributed or sorry about uh heterogeneous hardware?
2074.710–2074.720  sorry about uh heterogeneous hardware?
2074.720–2077.990  sorry about uh heterogeneous hardware? >> Uh, sure. Um, yes. Uh, basically on
2077.990–2078.000  >> Uh, sure. Um, yes. Uh, basically on
2078.000–2080.230  >> Uh, sure. Um, yes. Uh, basically on PyTorch, PyTorch specifically, we've
2080.230–2080.240  PyTorch, PyTorch specifically, we've
2080.240–2083.829  PyTorch, PyTorch specifically, we've been working on the project called CRCR,
2083.829–2083.839  been working on the project called CRCR,
2083.839–2087.030  been working on the project called CRCR, it's a cross repository CI relay. So, it
2087.030–2087.040  it's a cross repository CI relay. So, it
2087.040–2089.349  it's a cross repository CI relay. So, it lets partner back ends to run their own
2089.349–2089.359  lets partner back ends to run their own
2089.359–2092.470  lets partner back ends to run their own CI against PyTorch changes. So on the
2092.470–2092.480  CI against PyTorch changes. So on the
2092.480–2096.470  CI against PyTorch changes. So on the PyTorch PRS and report result back to us
2096.470–2096.480  PyTorch PRS and report result back to us
2096.480–2100.310  PyTorch PRS and report result back to us um on the PyTorch HUD page. Uh so we
2100.310–2100.320  um on the PyTorch HUD page. Uh so we
2100.320–2103.430  um on the PyTorch HUD page. Uh so we developed this with the partners um in
2103.430–2103.440  developed this with the partners um in
2103.440–2106.550  developed this with the partners um in the past few months and basically it's
2106.550–2106.560  the past few months and basically it's
2106.560–2109.349  the past few months and basically it's pretty much released right now for this
2109.349–2109.359  pretty much released right now for this
2109.359–2114.230  pretty much released right now for this um 2015 um 2014 release and this is a
2114.230–2114.240  um 2015 um 2014 release and this is a
2114.240–2116.310  um 2015 um 2014 release and this is a new feature in the Hut. So if you go to
2116.310–2116.320  new feature in the Hut. So if you go to
2116.320–2119.910  new feature in the Hut. So if you go to our hot page, there's the CRCR
2119.910–2119.920  our hot page, there's the CRCR
2119.920–2122.550  our hot page, there's the CRCR button. You can go and access this to
2122.550–2122.560  button. You can go and access this to
2122.560–2125.190  button. You can go and access this to have a like a preview. Uh so basically
2125.190–2125.200  have a like a preview. Uh so basically
2125.200–2127.990  have a like a preview. Uh so basically it's the way it's um configured. It's um
2127.990–2128.000  it's the way it's um configured. It's um
2128.000–2130.790  it's the way it's um configured. It's um the four levels of access. So level one
2130.790–2130.800  the four levels of access. So level one
2130.800–2133.670  the four levels of access. So level one is on boarding. So events are forwarded
2133.670–2133.680  is on boarding. So events are forwarded
2133.680–2138.150  is on boarding. So events are forwarded to downstream uh repositories and um
2138.150–2138.160  to downstream uh repositories and um
2138.160–2140.630  to downstream uh repositories and um upstream receive no feedback. So PyTorch
2140.630–2140.640  upstream receive no feedback. So PyTorch
2140.640–2143.430  upstream receive no feedback. So PyTorch we don't know what our
2143.430–2143.440  we don't know what our
2143.440–2146.790  we don't know what our tests are running on the downstream
2146.790–2146.800  tests are running on the downstream
2146.800–2150.150  tests are running on the downstream repos but tests are happening. So L2 is
2150.150–2150.160  repos but tests are happening. So L2 is
2150.160–2153.109  repos but tests are happening. So L2 is observation. It's when we actually uh
2153.109–2153.119  observation. It's when we actually uh
2153.119–2156.470  observation. It's when we actually uh PyTorch receives the results
2156.470–2156.480  PyTorch receives the results
2156.480–2160.069  PyTorch receives the results uh of the downstream um tests and able
2160.069–2160.079  uh of the downstream um tests and able
2160.079–2162.550  uh of the downstream um tests and able to visualize and see what are the
2162.550–2162.560  to visualize and see what are the
2162.560–2165.589  to visualize and see what are the failures and this helps like um uh
2165.589–2165.599  failures and this helps like um uh
2165.599–2171.190  failures and this helps like um uh observation uh and uh um L3 it's stable
2171.190–2171.200  observation uh and uh um L3 it's stable
2171.200–2175.109  observation uh and uh um L3 it's stable um it's where we are currently at um
2175.109–2175.119  um it's where we are currently at um
2175.119–2178.069  um it's where we are currently at um it's non-blocking run check on PR so
2178.069–2178.079  it's non-blocking run check on PR so
2178.079–2181.670  it's non-blocking run check on PR so it's basically you apply um flow out of
2181.670–2181.680  it's basically you apply um flow out of
2181.680–2185.030  it's basically you apply um flow out of tree lab out CR flow CRCR label to the
2185.030–2185.040  tree lab out CR flow CRCR label to the
2185.040–2187.829  tree lab out CR flow CRCR label to the PR and the results from the uh
2187.829–2187.839  PR and the results from the uh
2187.839–2190.150  PR and the results from the uh downstream repositories appear on your
2190.150–2190.160  downstream repositories appear on your
2190.160–2194.310  downstream repositories appear on your PyTorch PR and um
2194.310–2194.320  PyTorch PR and um
2194.320–2197.510  PyTorch PR and um currently we're working with Nindia
2197.510–2197.520  currently we're working with Nindia
2197.520–2201.990  currently we're working with Nindia Windowsci Red Hat IBM Spire Ascent NPU
2201.990–2202.000  Windowsci Red Hat IBM Spire Ascent NPU
2202.000–2206.055  Windowsci Red Hat IBM Spire Ascent NPU Google TPU and VLM repos for that. So
2206.055–2206.065  Google TPU and VLM repos for that. So
2206.065–2208.550  Google TPU and VLM repos for that. So [clears throat] this will help with um
2208.550–2208.560  [clears throat] this will help with um
2208.560–2211.109  [clears throat] this will help with um onboarding different um vendors,
2211.109–2211.119  onboarding different um vendors,
2211.119–2213.270  onboarding different um vendors, different architectures and different
2213.270–2213.280  different architectures and different
2213.280–2217.190  different architectures and different repositories uh and uh maintainers of
2217.190–2217.200  repositories uh and uh maintainers of
2217.200–2219.109  repositories uh and uh maintainers of those repositories should be able to
2219.109–2219.119  those repositories should be able to
2219.119–2222.069  those repositories should be able to onboard very quickly and get results
2222.069–2222.079  onboard very quickly and get results
2222.079–2225.510  onboard very quickly and get results testing for PyTorch pretty quickly and
2225.510–2225.520  testing for PyTorch pretty quickly and
2225.520–2228.069  testing for PyTorch pretty quickly and supported on PR testing and on nightly
2228.069–2228.079  supported on PR testing and on nightly
2228.079–2230.310  supported on PR testing and on nightly testing. So there are kind of two modes
2230.310–2230.320  testing. So there are kind of two modes
2230.320–2233.030  testing. So there are kind of two modes for this and yeah I think this is a very
2233.030–2233.040  for this and yeah I think this is a very
2233.040–2235.750  for this and yeah I think this is a very uh interesting uh part of the PyTorch
2235.750–2235.760  uh interesting uh part of the PyTorch
2235.760–2237.510  uh interesting uh part of the PyTorch project that we're currently developing
2237.510–2237.520  project that we're currently developing
2237.520–2240.069  project that we're currently developing and going to develop for
2240.069–2240.079  and going to develop for
2240.079–2241.589  and going to develop for um
2241.589–2241.599  um
2241.599–2242.870  um I don't know maybe Nikita
2242.870–2242.880  I don't know maybe Nikita
2242.880–2246.069  I don't know maybe Nikita >> yeah and maybe uh like one more yeah one
2246.069–2246.079  >> yeah and maybe uh like one more yeah one
2246.079–2247.670  >> yeah and maybe uh like one more yeah one more thing that like on the API side
2247.670–2247.680  more thing that like on the API side
2247.680–2249.430  more thing that like on the API side there's a torch accelerate API which
2249.430–2249.440  there's a torch accelerate API which
2249.440–2251.990  there's a torch accelerate API which tries to abstract away different devices
2251.990–2252.000  tries to abstract away different devices
2252.000–2254.230  tries to abstract away different devices different names different capabilities
2254.230–2254.240  different names different capabilities
2254.240–2256.710  different names different capabilities so you can write once for accelerated
2256.710–2256.720  so you can write once for accelerated
2256.720–2259.510  so you can write once for accelerated compute and it should on not I don't
2259.510–2259.520  compute and it should on not I don't
2259.520–2261.829  compute and it should on not I don't know I don't want to repeat the Java
2261.829–2261.839  know I don't want to repeat the Java
2261.839–2264.390  know I don't want to repeat the Java everywhere but it will run on multiple
2264.390–2264.400  everywhere but it will run on multiple
2264.400–2266.870  everywhere but it will run on multiple compute and we try to like align all our
2266.870–2266.880  compute and we try to like align all our
2266.880–2269.030  compute and we try to like align all our operator implementation this is why CRCR
2269.030–2269.040  operator implementation this is why CRCR
2269.040–2272.069  operator implementation this is why CRCR is there is that uh like vendors can
2272.069–2272.079  is there is that uh like vendors can
2272.079–2274.710  is there is that uh like vendors can make sure that like their hardware
2274.710–2274.720  make sure that like their hardware
2274.720–2278.470  make sure that like their hardware report the same results as we expect it
2278.470–2278.480  report the same results as we expect it
2278.480–2280.870  report the same results as we expect it would and developers will have fewer
2280.870–2280.880  would and developers will have fewer
2280.880–2282.950  would and developers will have fewer surprises when they use this like
2282.950–2282.960  surprises when they use this like
2282.960–2285.910  surprises when they use this like abstraction to dispatch their job to
2285.910–2285.920  abstraction to dispatch their job to
2285.920–2287.750  abstraction to dispatch their job to accelerator available but yes it is very
2287.750–2287.760  accelerator available but yes it is very
2287.760–2290.310  accelerator available but yes it is very important because you need to be on more
2290.310–2290.320  important because you need to be on more
2290.320–2293.510  important because you need to be on more platforms.
2295.109–2295.119  >> Yeah, I think I think I mean I I think
2295.119–2296.470  >> Yeah, I think I think I mean I I think this whole area is really fascinating
2296.470–2296.480  this whole area is really fascinating
2296.480–2300.230  this whole area is really fascinating within PyTorch because like uh like how
2300.230–2300.240  within PyTorch because like uh like how
2300.240–2302.310  within PyTorch because like uh like how do you how do you have like a consistent
2302.310–2302.320  do you how do you have like a consistent
2302.320–2303.990  do you how do you have like a consistent experience across the such different
2303.990–2304.000  experience across the such different
2304.000–2307.510  experience across the such different hardware and then how do we do QA? How
2307.510–2307.520  hardware and then how do we do QA? How
2307.520–2309.190  hardware and then how do we do QA? How do we deal with like new hardware
2309.190–2309.200  do we deal with like new hardware
2309.200–2310.950  do we deal with like new hardware emerging which may or may not be in the
2310.950–2310.960  emerging which may or may not be in the
2310.960–2313.030  emerging which may or may not be in the cloud for us to test? and how do we do
2313.030–2313.040  cloud for us to test? and how do we do
2313.040–2315.430  cloud for us to test? and how do we do like the the the the
2315.430–2315.440  like the the the the
2315.440–2318.069  like the the the the um the the the continuous integration
2318.069–2318.079  um the the the continuous integration
2318.079–2320.870  um the the the continuous integration testing in a distributed fashion. Um so
2320.870–2320.880  testing in a distributed fashion. Um so
2320.880–2323.349  testing in a distributed fashion. Um so so this is an area where like just a lot
2323.349–2323.359  so this is an area where like just a lot
2323.359–2324.630  so this is an area where like just a lot of effort has gone and I think we've
2324.630–2324.640  of effort has gone and I think we've
2324.640–2326.069  of effort has gone and I think we've just gotten like this part of PyTorch
2326.069–2326.079  just gotten like this part of PyTorch
2326.079–2327.589  just gotten like this part of PyTorch has gotten a lot more mature. So I
2327.589–2327.599  has gotten a lot more mature. So I
2327.599–2329.829  has gotten a lot more mature. So I imagine for hardware vendors it's way
2329.829–2329.839  imagine for hardware vendors it's way
2329.839–2332.069  imagine for hardware vendors it's way easier to sort of contemplate getting
2332.069–2332.079  easier to sort of contemplate getting
2332.079–2333.589  easier to sort of contemplate getting their their platform properly supported
2333.589–2333.599  their their platform properly supported
2333.599–2335.510  their their platform properly supported in PyTorch now than it was you know two
2335.510–2335.520  in PyTorch now than it was you know two
2335.520–2337.829  in PyTorch now than it was you know two years ago three years ago. So okay I
2337.829–2337.839  years ago three years ago. So okay I
2337.839–2338.790  years ago three years ago. So okay I think
2338.790–2338.800  think
2338.800–2341.270  think >> want to add sorry last last point to
2341.270–2341.280  >> want to add sorry last last point to
2341.280–2343.190  >> want to add sorry last last point to this and it's the way it's developed
2343.190–2343.200  this and it's the way it's developed
2343.200–2345.910  this and it's the way it's developed right now it's a lot more secure than it
2345.910–2345.920  right now it's a lot more secure than it
2345.920–2348.630  right now it's a lot more secure than it used to be. So there's no um sharing of
2348.630–2348.640  used to be. So there's no um sharing of
2348.640–2350.950  used to be. So there's no um sharing of the keys nothing like that. So on
2350.950–2350.960  the keys nothing like that. So on
2350.960–2353.109  the keys nothing like that. So on boarding and from infrastructure point
2353.109–2353.119  boarding and from infrastructure point
2353.119–2356.630  boarding and from infrastructure point of view it's very fast and secure and uh
2356.630–2356.640  of view it's very fast and secure and uh
2356.640–2358.550  of view it's very fast and secure and uh uh very straightforward. We set up
2358.550–2358.560  uh very straightforward. We set up
2358.560–2361.292  uh very straightforward. We set up specific repository CRC test when
2361.292–2361.302  specific repository CRC test when
2361.302–2362.390  specific repository CRC test when [clears throat] you can go and look at
2362.390–2362.400  [clears throat] you can go and look at
2362.400–2364.950  [clears throat] you can go and look at the on boarding guide and can start on
2364.950–2364.960  the on boarding guide and can start on
2364.960–2366.470  the on boarding guide and can start on boarding.
2366.470–2366.480  boarding.
2366.480–2370.390  boarding. >> Yeah. Awesome. Okay. I think we have a
2370.390–2370.400  >> Yeah. Awesome. Okay. I think we have a
2370.400–2373.750  >> Yeah. Awesome. Okay. I think we have a next question lined up.
2373.750–2373.760  next question lined up.
2373.760–2375.829  next question lined up. Okay. And this again thank you
2375.829–2375.839  Okay. And this again thank you
2375.839–2378.870  Okay. And this again thank you Shivashish uh for asking uh three
2378.870–2378.880  Shivashish uh for asking uh three
2378.880–2381.270  Shivashish uh for asking uh three interesting and challenging questions uh
2381.270–2381.280  interesting and challenging questions uh
2381.280–2383.510  interesting and challenging questions uh uh for us to answer here. Uh so for
2383.510–2383.520  uh for us to answer here. Uh so for
2383.520–2385.670  uh for us to answer here. Uh so for draft then verify loops uh which would
2385.670–2385.680  draft then verify loops uh which would
2385.680–2387.589  draft then verify loops uh which would be sort of speculative decoding style
2387.589–2387.599  be sort of speculative decoding style
2387.599–2391.030  be sort of speculative decoding style style is could a graph capture um in the
2391.030–2391.040  style is could a graph capture um in the
2391.040–2394.310  style is could a graph capture um in the uh torch while loop u uh mode usable on
2394.310–2394.320  uh torch while loop u uh mode usable on
2394.320–2396.630  uh torch while loop u uh mode usable on the verify step in 214 or does that
2396.630–2396.640  the verify step in 214 or does that
2396.640–2399.030  the verify step in 214 or does that control uh flow still force a graph
2399.030–2399.040  control uh flow still force a graph
2399.040–2403.109  control uh flow still force a graph break um do we do we know the answer to
2403.109–2403.119  break um do we do we know the answer to
2403.119–2404.390  break um do we do we know the answer to that among the
2404.390–2404.400  that among the
2404.400–2406.630  that among the >> I think generically you cannot answer
2406.630–2406.640  >> I think generically you cannot answer
2406.640–2408.310  >> I think generically you cannot answer this question because it depends on the
2408.310–2408.320  this question because it depends on the
2408.320–2411.510  this question because it depends on the workflow it should be usable and that's
2411.510–2411.520  workflow it should be usable and that's
2411.520–2414.550  workflow it should be usable and that's uh what the new um like abstractions
2414.550–2414.560  uh what the new um like abstractions
2414.560–2416.950  uh what the new um like abstractions that we talked about in the highlights
2416.950–2416.960  that we talked about in the highlights
2416.960–2418.790  that we talked about in the highlights are for but there could be some cases
2418.790–2418.800  are for but there could be some cases
2418.800–2421.589  are for but there could be some cases when it's not but we you already have uh
2421.589–2421.599  when it's not but we you already have uh
2421.599–2424.630  when it's not but we you already have uh toggles to test that you can just wrap
2424.630–2424.640  toggles to test that you can just wrap
2424.640–2427.430  toggles to test that you can just wrap your code with like aborton graph break
2427.430–2427.440  your code with like aborton graph break
2427.440–2429.430  your code with like aborton graph break and then you know that you produced a
2429.430–2429.440  and then you know that you produced a
2429.440–2431.829  and then you know that you produced a graph break or not but it's kind of I I
2431.829–2431.839  graph break or not but it's kind of I I
2431.839–2433.349  graph break or not but it's kind of I I don't think there's one one answer that
2433.349–2433.359  don't think there's one one answer that
2433.359–2436.630  don't think there's one one answer that fits them all like we try to make those
2436.630–2436.640  fits them all like we try to make those
2436.640–2439.270  fits them all like we try to make those not cause graph breaks but depending on
2439.270–2439.280  not cause graph breaks but depending on
2439.280–2441.030  not cause graph breaks but depending on the CUDA version you're on the hardware
2441.030–2441.040  the CUDA version you're on the hardware
2441.040–2443.190  the CUDA version you're on the hardware capabilities is a Python version and so
2443.190–2443.200  capabilities is a Python version and so
2443.200–2444.870  capabilities is a Python version and so on and so forth. It can cause graph
2444.870–2444.880  on and so forth. It can cause graph
2444.880–2448.790  on and so forth. It can cause graph break. So please check if it doesn't
2448.790–2448.800  break. So please check if it doesn't
2448.800–2451.109  break. So please check if it doesn't create great if it does file an issue
2451.109–2451.119  create great if it does file an issue
2451.119–2453.430  create great if it does file an issue but we try to eliminate as graph breaks
2453.430–2453.440  but we try to eliminate as graph breaks
2453.440–2455.510  but we try to eliminate as graph breaks as much as possible because it slows
2455.510–2455.520  as much as possible because it slows
2455.520–2457.190  as much as possible because it slows down
2457.190–2457.200  down
2457.200–2459.829  down your pipelines.
2459.829–2459.839  your pipelines.
2459.839–2462.710  your pipelines. >> Okay, awesome. Um, can we jump next to
2462.710–2462.720  >> Okay, awesome. Um, can we jump next to
2462.720–2467.030  >> Okay, awesome. Um, can we jump next to the question about Envy Gym uh Q5? Uh,
2467.030–2467.040  the question about Envy Gym uh Q5? Uh,
2467.040–2471.589  the question about Envy Gym uh Q5? Uh, Basil,
2473.349–2473.359  >> sorry to do that verbally rather than in
2473.359–2483.430  >> sorry to do that verbally rather than in the chat.
2485.349–2485.359  So, this is one of the key things that
2485.359–2486.710  So, this is one of the key things that we highlighted at the be at the
2486.710–2486.720  we highlighted at the be at the
2486.720–2487.829  we highlighted at the be at the beginning and I wanted to make sure we
2487.829–2487.839  beginning and I wanted to make sure we
2487.839–2489.510  beginning and I wanted to make sure we we took a moment to talk about it among
2489.510–2489.520  we took a moment to talk about it among
2489.520–2492.710  we took a moment to talk about it among the questions. Uh so what is NVGEM? Um
2492.710–2492.720  the questions. Uh so what is NVGEM? Um
2492.720–2494.309  the questions. Uh so what is NVGEM? Um kind of a generic question and how do I
2494.309–2494.319  kind of a generic question and how do I
2494.319–2498.630  kind of a generic question and how do I turn it on? Um uh so uh Nikita do you
2498.630–2498.640  turn it on? Um uh so uh Nikita do you
2498.640–2500.550  turn it on? Um uh so uh Nikita do you want to jump on this or Andre?
2500.550–2500.560  want to jump on this or Andre?
2500.560–2502.950  want to jump on this or Andre? >> Sure. But we already somewhat covered it
2502.950–2502.960  >> Sure. But we already somewhat covered it
2502.960–2504.550  >> Sure. But we already somewhat covered it when the other user asked a question
2504.550–2504.560  when the other user asked a question
2504.560–2508.230  when the other user asked a question about different uh good gen backends and
2508.230–2508.240  about different uh good gen backends and
2508.240–2510.390  about different uh good gen backends and I said that there is a like you can't do
2510.390–2510.400  I said that there is a like you can't do
2510.400–2512.710  I said that there is a like you can't do it over FX graph but there is a toggles
2512.710–2512.720  it over FX graph but there is a toggles
2512.720–2516.309  it over FX graph but there is a toggles and specifically for NVJ there's max to
2516.309–2516.319  and specifically for NVJ there's max to
2516.319–2517.910  and specifically for NVJ there's max to tune gem backends
2517.910–2517.920  tune gem backends
2517.920–2520.390  tune gem backends >> list and if you add NVJM here so it's
2520.390–2520.400  >> list and if you add NVJM here so it's
2520.400–2522.470  >> list and if you add NVJM here so it's not enabled by default but then it will
2522.470–2522.480  not enabled by default but then it will
2522.480–2524.470  not enabled by default but then it will be enabled and it will allow some more
2524.470–2524.480  be enabled and it will allow some more
2524.480–2528.069  be enabled and it will allow some more advanced epilog fusion and yeah NVJM is
2528.069–2528.079  advanced epilog fusion and yeah NVJM is
2528.079–2532.309  advanced epilog fusion and yeah NVJM is um QGSL gem back end that based on
2532.309–2532.319  um QGSL gem back end that based on
2532.319–2534.230  um QGSL gem back end that based on cutless
2534.230–2534.240  cutless
2534.240–2537.109  cutless and it allows you to like yeah fuse
2537.109–2537.119  and it allows you to like yeah fuse
2537.119–2539.270  and it allows you to like yeah fuse mules and fusions and get generate even
2539.270–2539.280  mules and fusions and get generate even
2539.280–2543.349  mules and fusions and get generate even faster uh yeah uh kernels specifically
2543.349–2543.359  faster uh yeah uh kernels specifically
2543.359–2545.109  faster uh yeah uh kernels specifically if you do low precision data types like
2545.109–2545.119  if you do low precision data types like
2545.119–2547.589  if you do low precision data types like in VFP4 but it requires you to have an
2547.589–2547.599  in VFP4 but it requires you to have an
2547.599–2550.069  in VFP4 but it requires you to have an Nvidia cutless DSL installed on your
2550.069–2550.079  Nvidia cutless DSL installed on your
2550.079–2553.670  Nvidia cutless DSL installed on your system and blackwell as well so like you
2553.670–2553.680  system and blackwell as well so like you
2553.680–2555.910  system and blackwell as well so like you unfortunately cannot use 44 [laughter]
2555.910–2555.920  unfortunately cannot use 44 [laughter]
2555.920–2562.230  unfortunately cannot use 44 [laughter] on your ampers or hoppers
2565.670–2565.680  Okay, I see a question from Jose Silva
2565.680–2570.150  Okay, I see a question from Jose Silva on the on the chat. Um,
2570.150–2570.160  on the on the chat. Um,
2570.160–2572.870  on the on the chat. Um, okay, awesome. Thank you, Basil. Um, if
2572.870–2572.880  okay, awesome. Thank you, Basil. Um, if
2572.880–2575.349  okay, awesome. Thank you, Basil. Um, if I train a vision model on CUDA and
2575.349–2575.359  I train a vision model on CUDA and
2575.359–2578.309  I train a vision model on CUDA and export it using AOT inductor to C++
2578.309–2578.319  export it using AOT inductor to C++
2578.319–2581.750  export it using AOT inductor to C++ binaries for a machine running an Intel
2581.750–2581.760  binaries for a machine running an Intel
2581.760–2585.190  binaries for a machine running an Intel Iris iGPU. Um, is dynamic shape and
2585.190–2585.200  Iris iGPU. Um, is dynamic shape and
2585.200–2586.950  Iris iGPU. Um, is dynamic shape and anti-quantization supported out of the
2586.950–2586.960  anti-quantization supported out of the
2586.960–2589.589  anti-quantization supported out of the box on those targets? On those XPU
2589.589–2589.599  box on those targets? On those XPU
2589.599–2591.854  box on those targets? On those XPU targets?
2591.854–2591.864  targets?
2591.864–2593.030  targets? [laughter]
2593.030–2593.040  [laughter]
2593.040–2594.870  [laughter] >> That's quite a specific question.
2594.870–2594.880  >> That's quite a specific question.
2594.880–2596.390  >> That's quite a specific question. >> I was just going to say I don't know
2596.390–2596.400  >> I was just going to say I don't know
2596.400–2599.059  >> I was just going to say I don't know that case is covered in rci. Uh,
2599.059–2599.069  that case is covered in rci. Uh,
2599.069–2600.150  that case is covered in rci. Uh, [laughter]
2600.150–2600.160  [laughter]
2600.160–2602.470  [laughter] >> it might be covered in rci, but maybe
2602.470–2602.480  >> it might be covered in rci, but maybe
2602.480–2605.510  >> it might be covered in rci, but maybe not intel Iris iGPU in particular. Like
2605.510–2605.520  not intel Iris iGPU in particular. Like
2605.520–2608.069  not intel Iris iGPU in particular. Like I think they're trying to cover a more
2608.069–2608.079  I think they're trying to cover a more
2608.079–2611.430  I think they're trying to cover a more >> modern GPUs. Uh
2611.430–2611.440  >> modern GPUs. Uh
2611.440–2615.349  >> modern GPUs. Uh but I guess you never know until you try
2615.349–2615.359  but I guess you never know until you try
2615.359–2617.910  but I guess you never know until you try and then you can probably file an issue
2617.910–2617.920  and then you can probably file an issue
2617.920–2618.870  and then you can probably file an issue if it doesn't work.
2618.870–2618.880  if it doesn't work.
2618.880–2619.750  if it doesn't work. >> Yeah, file is
2619.750–2619.760  >> Yeah, file is
2619.760–2622.230  >> Yeah, file is >> exactly [laughter]
2622.950–2622.960  >> help us out.
2622.960–2624.950  >> help us out. >> And we do have Yeah, we do have support
2624.950–2624.960  >> And we do have Yeah, we do have support
2624.960–2627.910  >> And we do have Yeah, we do have support for XPU devices. So on both PyTorch and
2627.910–2627.920  for XPU devices. So on both PyTorch and
2627.920–2631.510  for XPU devices. So on both PyTorch and torch vision. So uh if you have a PR in
2631.510–2631.520  torch vision. So uh if you have a PR in
2631.520–2634.150  torch vision. So uh if you have a PR in mind, you can go and open a PR against
2634.150–2634.160  mind, you can go and open a PR against
2634.160–2637.829  mind, you can go and open a PR against PyTorch Vision CI.
2637.829–2637.839  PyTorch Vision CI.
2637.839–2640.230  PyTorch Vision CI. But like I guess if I read the question
2640.230–2640.240  But like I guess if I read the question
2640.240–2642.309  But like I guess if I read the question correctly, if you train model on CUDA,
2642.309–2642.319  correctly, if you train model on CUDA,
2642.319–2644.470  correctly, if you train model on CUDA, can you export it for XPU? You need to
2644.470–2644.480  can you export it for XPU? You need to
2644.480–2646.069  can you export it for XPU? You need to make some changes to your model first,
2646.069–2646.079  make some changes to your model first,
2646.079–2648.230  make some changes to your model first, right? You need to like transfer your
2648.230–2648.240  right? You need to like transfer your
2648.240–2650.470  right? You need to like transfer your model to XPU, make sure that it works
2650.470–2650.480  model to XPU, make sure that it works
2650.480–2653.750  model to XPU, make sure that it works and then AOT inductor should work on XPU
2653.750–2653.760  and then AOT inductor should work on XPU
2653.760–2655.589  and then AOT inductor should work on XPU same as it works on any other back ends.
2655.589–2655.599  same as it works on any other back ends.
2655.599–2658.470  same as it works on any other back ends. But I'm not even sure if int 8
2658.470–2658.480  But I'm not even sure if int 8
2658.480–2660.069  But I'm not even sure if int 8 contisation is supported through
2660.069–2660.079  contisation is supported through
2660.079–2664.390  contisation is supported through inductor even for CUDA. like I think
2664.390–2664.400  inductor even for CUDA. like I think
2664.400–2665.990  inductor even for CUDA. like I think that there are lots of like layers to
2665.990–2666.000  that there are lots of like layers to
2666.000–2668.870  that there are lots of like layers to unload here. So please file an issue but
2668.870–2668.880  unload here. So please file an issue but
2668.880–2672.550  unload here. So please file an issue but I also kind of think why do you want an
2672.550–2672.560  I also kind of think why do you want an
2672.560–2678.309  I also kind of think why do you want an int 8 rather than like well int 8 and
2678.309–2678.319  int 8 rather than like well int 8 and
2678.319–2680.390  int 8 rather than like well int 8 and and check the spec if iGPU even support
2680.390–2680.400  and check the spec if iGPU even support
2680.400–2683.190  and check the spec if iGPU even support an int8ate uh fast smart n I know that
2683.190–2683.200  an int8ate uh fast smart n I know that
2683.200–2685.750  an int8ate uh fast smart n I know that CPUs do but I'm not sure about GPUs in
2685.750–2685.760  CPUs do but I'm not sure about GPUs in
2685.760–2690.309  CPUs do but I'm not sure about GPUs in particular.
2697.030–2697.040  Um
2697.040–2701.829  Um I think Basil's pulling the one that we
2701.829–2701.839  I think Basil's pulling the one that we
2701.839–2705.910  I think Basil's pulling the one that we mentioned to him here.
2709.510–2709.520  Maybe in the meantime we can do a small
2709.520–2712.870  Maybe in the meantime we can do a small uh note about PyTorch release matrix. So
2712.870–2712.880  uh note about PyTorch release matrix. So
2712.880–2715.510  uh note about PyTorch release matrix. So one of the changes that is in 214 is
2715.510–2715.520  one of the changes that is in 214 is
2715.520–2717.349  one of the changes that is in 214 is that we mandate that all of your
2717.349–2717.359  that we mandate that all of your
2717.359–2719.190  that we mandate that all of your extensions will be compiled with oh here
2719.190–2719.200  extensions will be compiled with oh here
2719.200–2721.510  extensions will be compiled with oh here you go C++ 20 by default. So if you're
2721.510–2721.520  you go C++ 20 by default. So if you're
2721.520–2723.589  you go C++ 20 by default. So if you're using an older compiler, you really need
2723.589–2723.599  using an older compiler, you really need
2723.599–2726.390  using an older compiler, you really need to update. It is 2026 and majority of
2726.390–2726.400  to update. It is 2026 and majority of
2726.400–2728.390  to update. It is 2026 and majority of the compiler should be able to support
2728.390–2728.400  the compiler should be able to support
2728.400–2732.150  the compiler should be able to support C++ 20. Uh but that's
2732.150–2732.160  C++ 20. Uh but that's
2732.160–2733.670  C++ 20. Uh but that's backward compatibility breaking change.
2733.670–2733.680  backward compatibility breaking change.
2733.680–2735.349  backward compatibility breaking change. And with that, I don't know, Andrew, do
2735.349–2735.359  And with that, I don't know, Andrew, do
2735.359–2736.710  And with that, I don't know, Andrew, do you want to Oh, Chris, do you want to
2736.710–2736.720  you want to Oh, Chris, do you want to
2736.720–2737.750  you want to Oh, Chris, do you want to read the question first?
2737.750–2737.760  read the question first?
2737.760–2739.270  read the question first? >> Yeah. Yeah. Yeah. So, so this this
2739.270–2739.280  >> Yeah. Yeah. Yeah. So, so this this
2739.280–2740.550  >> Yeah. Yeah. Yeah. So, so this this question which I think was going in the
2740.550–2740.560  question which I think was going in the
2740.560–2742.790  question which I think was going in the same direction that that Nikita was was
2742.790–2742.800  same direction that that Nikita was was
2742.800–2744.950  same direction that that Nikita was was thinking is do I still have to upgrade
2744.950–2744.960  thinking is do I still have to upgrade
2744.960–2747.109  thinking is do I still have to upgrade um torch vision every time I upgrade
2747.109–2747.119  um torch vision every time I upgrade
2747.119–2748.309  um torch vision every time I upgrade grade pietorch which I know was a
2748.309–2748.319  grade pietorch which I know was a
2748.319–2749.990  grade pietorch which I know was a headache that people uh had to deal with
2749.990–2750.000  headache that people uh had to deal with
2750.000–2752.790  headache that people uh had to deal with for a very long time that um Andre do
2752.790–2752.800  for a very long time that um Andre do
2752.800–2754.069  for a very long time that um Andre do you want to respond to this?
2754.069–2754.079  you want to respond to this?
2754.079–2758.150  you want to respond to this? >> Yes. Yes. Um so torch vision uh 0.29 is
2758.150–2758.160  >> Yes. Yes. Um so torch vision uh 0.29 is
2758.160–2763.510  >> Yes. Yes. Um so torch vision uh 0.29 is a stable uh with respect to 21.14. So it
2763.510–2763.520  a stable uh with respect to 21.14. So it
2763.520–2765.430  a stable uh with respect to 21.14. So it will keep working with later version
2765.430–2765.440  will keep working with later version
2765.440–2769.109  will keep working with later version 2.15 2.16 and later. Uh the only catch
2769.109–2769.119  2.15 2.16 and later. Uh the only catch
2769.119–2771.349  2.15 2.16 and later. Uh the only catch here if you need to um introduce new
2771.349–2771.359  here if you need to um introduce new
2771.359–2774.950  here if you need to um introduce new CUDA version you may need to download
2774.950–2774.960  CUDA version you may need to download
2774.960–2777.829  CUDA version you may need to download um version of torch vision compiled with
2777.829–2777.839  um version of torch vision compiled with
2777.839–2782.069  um version of torch vision compiled with this CUDA version. Um but yes um like as
2782.069–2782.079  this CUDA version. Um but yes um like as
2782.079–2783.910  this CUDA version. Um but yes um like as um
2783.910–2783.920  um
2783.920–2786.710  um as a basic you no longer need matching
2786.710–2786.720  as a basic you no longer need matching
2786.720–2788.870  as a basic you no longer need matching torch vision to install every torch
2788.870–2788.880  torch vision to install every torch
2788.880–2793.190  torch vision to install every torch upgrade and uh in the future we may not
2793.190–2793.200  upgrade and uh in the future we may not
2793.200–2795.349  upgrade and uh in the future we may not do torch vision at the same time as
2795.349–2795.359  do torch vision at the same time as
2795.359–2799.030  do torch vision at the same time as torch. So the releases could be um a
2799.030–2799.040  torch. So the releases could be um a
2799.040–2802.069  torch. So the releases could be um a little bit decoupled because of this. So
2802.069–2802.079  little bit decoupled because of this. So
2802.079–2804.630  little bit decoupled because of this. So I think it's something good for us. It's
2804.630–2804.640  I think it's something good for us. It's
2804.640–2808.950  I think it's something good for us. It's a good news.
2810.710–2810.720  >> Just next next question, please.
2810.720–2813.829  >> Just next next question, please. >> Next question.
2813.829–2813.839  >> Next question.
2813.839–2816.470  >> Next question. >> Okay. So, what is new for writing
2816.470–2816.480  >> Okay. So, what is new for writing
2816.480–2820.309  >> Okay. So, what is new for writing dynamic models that still compile? Well,
2820.309–2820.319  dynamic models that still compile? Well,
2820.319–2823.270  dynamic models that still compile? Well, um Andre, do you want to take this one
2823.270–2823.280  um Andre, do you want to take this one
2823.280–2826.390  um Andre, do you want to take this one as as as well?
2826.390–2826.400  as as as well?
2826.400–2828.470  as as as well? >> I think we answered this question kind
2828.470–2828.480  >> I think we answered this question kind
2828.480–2830.069  >> I think we answered this question kind of twice already. There was a question
2830.069–2830.079  of twice already. There was a question
2830.079–2831.829  of twice already. There was a question separate. Like if we want to highlight
2831.829–2831.839  separate. Like if we want to highlight
2831.839–2834.710  separate. Like if we want to highlight the features there is a torch switch uh
2834.710–2834.720  the features there is a torch switch uh
2834.720–2836.470  the features there is a torch switch uh torch while loop that is captured across
2836.470–2836.480  torch while loop that is captured across
2836.480–2839.589  torch while loop that is captured across codraph and dynamic spec and people
2839.589–2839.599  codraph and dynamic spec and people
2839.599–2841.190  codraph and dynamic spec and people already asked about dynamic spec and
2841.190–2841.200  already asked about dynamic spec and
2841.200–2844.069  already asked about dynamic spec and whether it will cause inductor to fail
2844.069–2844.079  whether it will cause inductor to fail
2844.079–2845.750  whether it will cause inductor to fail when it goes above like an upper
2845.750–2845.760  when it goes above like an upper
2845.760–2848.230  when it goes above like an upper boundary. People talked about uh asked
2848.230–2848.240  boundary. People talked about uh asked
2848.240–2849.829  boundary. People talked about uh asked about specialization but yes those are
2849.829–2849.839  about specialization but yes those are
2849.839–2852.790  about specialization but yes those are the features that you should try and
2852.790–2852.800  the features that you should try and
2852.800–2855.990  the features that you should try and like uh they will improve your VSC to
2855.990–2856.000  like uh they will improve your VSC to
2856.000–2858.950  like uh they will improve your VSC to compile model with dynamically shaped
2858.950–2858.960  compile model with dynamically shaped
2858.960–2861.750  compile model with dynamically shaped inputs like you will have fun um
2861.750–2861.760  inputs like you will have fun um
2861.760–2863.430  inputs like you will have fun um compiled artifacts that works for
2863.430–2863.440  compiled artifacts that works for
2863.440–2865.750  compiled artifacts that works for multiple shapes. So once again it's in
2865.750–2865.760  multiple shapes. So once again it's in
2865.760–2868.870  multiple shapes. So once again it's in the torch switch [laughter] torch loop
2868.870–2868.880  the torch switch [laughter] torch loop
2868.880–2871.750  the torch switch [laughter] torch loop and dynamics back.
2871.750–2871.760  and dynamics back.
2871.760–2874.630  and dynamics back. >> Uh can we jump to question 14 in our
2874.630–2874.640  >> Uh can we jump to question 14 in our
2874.640–2878.150  >> Uh can we jump to question 14 in our prepared ones? Um
2878.150–2878.160  prepared ones? Um
2878.160–2882.230  prepared ones? Um this is the everpresent Torch script uh
2882.230–2882.240  this is the everpresent Torch script uh
2882.240–2884.150  this is the everpresent Torch script uh topic [snorts]
2884.150–2884.160  topic [snorts]
2884.160–2885.430  topic [snorts] trying to touch on things that we
2885.430–2885.440  trying to touch on things that we
2885.440–2887.750  trying to touch on things that we haven't already discussed. Okay. So I
2887.750–2887.760  haven't already discussed. Okay. So I
2887.760–2889.430  haven't already discussed. Okay. So I still use Torch script. Should I be
2889.430–2889.440  still use Torch script. Should I be
2889.440–2891.910  still use Torch script. Should I be worried? Um Nikita, do you want to
2891.910–2891.920  worried? Um Nikita, do you want to
2891.920–2893.510  worried? Um Nikita, do you want to answer this one?
2893.510–2893.520  answer this one?
2893.520–2896.230  answer this one? Yes, we've been saying that to script
2896.230–2896.240  Yes, we've been saying that to script
2896.240–2900.550  Yes, we've been saying that to script has been deprecated since I think 211 or
2900.550–2900.560  has been deprecated since I think 211 or
2900.560–2903.750  has been deprecated since I think 211 or 212 and it has not been really working
2903.750–2903.760  212 and it has not been really working
2903.760–2905.990  212 and it has not been really working even like you cannot use script on you
2905.990–2906.000  even like you cannot use script on you
2906.000–2908.230  even like you cannot use script on you cannot trace the store script on Python
2908.230–2908.240  cannot trace the store script on Python
2908.240–2911.109  cannot trace the store script on Python 314 and I'm not sure what is the status
2911.109–2911.119  314 and I'm not sure what is the status
2911.119–2913.990  314 and I'm not sure what is the status of 312 or 313 like it's just being
2913.990–2914.000  of 312 or 313 like it's just being
2914.000–2915.750  of 312 or 313 like it's just being deprecated there are bugs there are
2915.750–2915.760  deprecated there are bugs there are
2915.760–2918.230  deprecated there are bugs there are security problems it supports less and
2918.230–2918.240  security problems it supports less and
2918.240–2919.990  security problems it supports less and less operators I don't think you can
2919.990–2920.000  less operators I don't think you can
2920.000–2923.190  less operators I don't think you can capture anything that support
2923.190–2923.200  capture anything that support
2923.200–2925.270  capture anything that support flotate or whatever. So don't use to
2925.270–2925.280  flotate or whatever. So don't use to
2925.280–2928.069  flotate or whatever. So don't use to script anymore. So AOTI expert is a
2928.069–2928.079  script anymore. So AOTI expert is a
2928.079–2932.470  script anymore. So AOTI expert is a pretty uh like mature framework at this
2932.470–2932.480  pretty uh like mature framework at this
2932.480–2934.390  pretty uh like mature framework at this point. Use this one. If you need to use
2934.390–2934.400  point. Use this one. If you need to use
2934.400–2936.230  point. Use this one. If you need to use script, you can use an older PyTorch but
2936.230–2936.240  script, you can use an older PyTorch but
2936.240–2938.470  script, you can use an older PyTorch but like be aware of the limitation. Be like
2938.470–2938.480  like be aware of the limitation. Be like
2938.480–2940.630  like be aware of the limitation. Be like we will be happy if there is a community
2940.630–2940.640  we will be happy if there is a community
2940.640–2943.190  we will be happy if there is a community maintainer who want to like look into
2943.190–2943.200  maintainer who want to like look into
2943.200–2946.069  maintainer who want to like look into the torch script but we are not actively
2946.069–2946.079  the torch script but we are not actively
2946.079–2948.309  the torch script but we are not actively working on new features or have anyone
2948.309–2948.319  working on new features or have anyone
2948.319–2950.230  working on new features or have anyone who actively like triages in common
2950.230–2950.240  who actively like triages in common
2950.240–2952.549  who actively like triages in common issues. So unless they are security
2952.549–2952.559  issues. So unless they are security
2952.559–2956.950  issues. So unless they are security related they will probably stay as is
2956.950–2956.960  related they will probably stay as is
2956.960–2958.950  related they will probably stay as is and craft and script is not a security
2958.950–2958.960  and craft and script is not a security
2958.960–2962.790  and craft and script is not a security issue unfortunately.
2965.270–2965.280  >> Okay we have I think we have one left
2965.280–2969.190  >> Okay we have I think we have one left one more remaining um prepared question.
2969.190–2969.200  one more remaining um prepared question.
2969.200–2970.950  one more remaining um prepared question. Um
2970.950–2970.960  Um
2970.960–2973.430  Um >> we switch to community ones or whatever.
2973.430–2973.440  >> we switch to community ones or whatever.
2973.440–2975.270  >> we switch to community ones or whatever. >> No I
2975.270–2975.280  >> No I
2975.280–2976.549  >> No I okay I think we have one more prepared
2976.549–2976.559  okay I think we have one more prepared
2976.559–2977.990  okay I think we have one more prepared one. If there are community ones let's
2977.990–2978.000  one. If there are community ones let's
2978.000–2979.430  one. If there are community ones let's go back and make sure we catch them
2979.430–2979.440  go back and make sure we catch them
2979.440–2982.150  go back and make sure we catch them before we end. Um, so I hear that uh
2982.150–2982.160  before we end. Um, so I hear that uh
2982.160–2986.069  before we end. Um, so I hear that uh 21.14 supports Python 315. Can I just
2986.069–2986.079  21.14 supports Python 315. Can I just
2986.079–2988.470  21.14 supports Python 315. Can I just pip install Torch on it? Um, Andre, do
2988.470–2988.480  pip install Torch on it? Um, Andre, do
2988.480–2989.910  pip install Torch on it? Um, Andre, do you want to take this one?
2989.910–2989.920  you want to take this one?
2989.920–2992.309  you want to take this one? >> Um, yes. Uh, so basically the short
2992.309–2992.319  >> Um, yes. Uh, so basically the short
2992.319–2995.990  >> Um, yes. Uh, so basically the short answer, we do support 315. Uh, but it's
2995.990–2996.000  answer, we do support 315. Uh, but it's
2996.000–2999.109  answer, we do support 315. Uh, but it's only available via download pytorch.org.
2999.109–2999.119  only available via download pytorch.org.
2999.119–3002.630  only available via download pytorch.org. So pip install torch will not work with
3002.630–3002.640  So pip install torch will not work with
3002.640–3007.349  So pip install torch will not work with 315 for this release 2014. Um
3007.349–3007.359  315 for this release 2014. Um
3007.359–3009.910  315 for this release 2014. Um having said that in the future uh we are
3009.910–3009.920  having said that in the future uh we are
3009.920–3013.510  having said that in the future uh we are planning on um uh publishing for the
3013.510–3013.520  planning on um uh publishing for the
3013.520–3016.710  planning on um uh publishing for the next release 2.15 we're publishing uh
3016.710–3016.720  next release 2.15 we're publishing uh
3016.720–3021.349  next release 2.15 we're publishing uh 315 Python to the pipy as a default one.
3021.349–3021.359  315 Python to the pipy as a default one.
3021.359–3025.030  315 Python to the pipy as a default one. We also be duplicating Python 3.10 for
3025.030–3025.040  We also be duplicating Python 3.10 for
3025.040–3026.630  We also be duplicating Python 3.10 for the next release. I think it's very
3026.630–3026.640  the next release. I think it's very
3026.640–3029.829  the next release. I think it's very important if you're still running 3.10
3029.829–3029.839  important if you're still running 3.10
3029.839–3031.829  important if you're still running 3.10 for next release it will not be
3031.829–3031.839  for next release it will not be
3031.839–3035.349  for next release it will not be supported. no wheels published and um
3035.349–3035.359  supported. no wheels published and um
3035.359–3037.349  supported. no wheels published and um compiling from source will not be
3037.349–3037.359  compiling from source will not be
3037.359–3040.069  compiling from source will not be possible. So we'll be migrating or
3040.069–3040.079  possible. So we'll be migrating or
3040.079–3042.710  possible. So we'll be migrating or supporting minimum 311.
3042.710–3042.720  supporting minimum 311.
3042.720–3046.470  supporting minimum 311. Um but yes for the current version 310
3046.470–3046.480  Um but yes for the current version 310
3046.480–3051.109  Um but yes for the current version 310 3.15 including uh free threaded 3.15T is
3051.109–3051.119  3.15 including uh free threaded 3.15T is
3051.119–3054.069  3.15 including uh free threaded 3.15T is are supported. However, one more call
3054.069–3054.079  are supported. However, one more call
3054.079–3056.390  are supported. However, one more call out is that torch compile is still not
3056.390–3056.400  out is that torch compile is still not
3056.400–3060.230  out is that torch compile is still not enabled for 315 and 315T.
3060.230–3060.240  enabled for 315 and 315T.
3060.240–3063.829  enabled for 315 and 315T. Um yes I think that's that's about uh
3063.829–3063.839  Um yes I think that's that's about uh
3063.839–3066.470  Um yes I think that's that's about uh covers it. I don't know if anybody else
3066.470–3066.480  covers it. I don't know if anybody else
3066.480–3070.710  covers it. I don't know if anybody else have any additional thoughts.
3070.710–3070.720  have any additional thoughts.
3070.720–3073.109  have any additional thoughts. Oh, maybe like kind of a reason why we
3073.109–3073.119  Oh, maybe like kind of a reason why we
3073.119–3075.670  Oh, maybe like kind of a reason why we don't support 310 anymore because uh it
3075.670–3075.680  don't support 310 anymore because uh it
3075.680–3079.030  don't support 310 anymore because uh it reached its end of life per um like uh
3079.030–3079.040  reached its end of life per um like uh
3079.040–3081.190  reached its end of life per um like uh Python end of life policy for the
3081.190–3081.200  Python end of life policy for the
3081.200–3084.549  Python end of life policy for the releases and we also will be like so we
3084.549–3084.559  releases and we also will be like so we
3084.559–3086.390  releases and we also will be like so we constantly reviewing the supported
3086.390–3086.400  constantly reviewing the supported
3086.400–3088.390  constantly reviewing the supported platforms and unfortunately retires the
3088.390–3088.400  platforms and unfortunately retires the
3088.400–3091.670  platforms and unfortunately retires the ones that like go out of support so we
3091.670–3091.680  ones that like go out of support so we
3091.680–3093.109  ones that like go out of support so we probably also be stopping supporting
3093.109–3093.119  probably also be stopping supporting
3093.119–3095.750  probably also be stopping supporting like Mac OS I forget what was released
3095.750–3095.760  like Mac OS I forget what was released
3095.760–3097.829  like Mac OS I forget what was released from three years ago but the typical and
3097.829–3097.839  from three years ago but the typical and
3097.839–3099.670  from three years ago but the typical and life cycle for those releases are also
3099.670–3099.680  life cycle for those releases are also
3099.680–3102.470  life cycle for those releases are also three So we will be returning that one
3102.470–3102.480  three So we will be returning that one
3102.480–3105.589  three So we will be returning that one as well. And again CUDA 12 that was
3105.589–3105.599  as well. And again CUDA 12 that was
3105.599–3107.270  as well. And again CUDA 12 that was mentioned before. Let's repeat it again
3107.270–3107.280  mentioned before. Let's repeat it again
3107.280–3109.270  mentioned before. Let's repeat it again that CUDA 12 support is going away. So
3109.270–3109.280  that CUDA 12 support is going away. So
3109.280–3110.630  that CUDA 12 support is going away. So you should still be able to build from
3110.630–3110.640  you should still be able to build from
3110.640–3113.270  you should still be able to build from source. We try not to break that thing.
3113.270–3113.280  source. We try not to break that thing.
3113.280–3115.270  source. We try not to break that thing. And I don't think we will actively kind
3115.270–3115.280  And I don't think we will actively kind
3115.280–3118.309  And I don't think we will actively kind of break 310 but we will not be done
3118.309–3118.319  of break 310 but we will not be done
3118.319–3121.349  of break 310 but we will not be done like testing it. So it will get broken
3121.349–3121.359  like testing it. So it will get broken
3121.359–3123.270  like testing it. So it will get broken eventually.
3123.270–3123.280  eventually.
3123.280–3124.630  eventually. Cool.
3124.630–3124.640  Cool.
3124.640–3126.390  Cool. >> Okay. I think we have a couple more
3126.390–3126.400  >> Okay. I think we have a couple more
3126.400–3129.910  >> Okay. I think we have a couple more questions lined up. Um, so, uh, Brian
3129.910–3129.920  questions lined up. Um, so, uh, Brian
3129.920–3131.990  questions lined up. Um, so, uh, Brian asks, "How can I use PyTorch to create
3131.990–3132.000  asks, "How can I use PyTorch to create
3132.000–3134.069  asks, "How can I use PyTorch to create my own LLMs?"
3134.069–3134.079  my own LLMs?"
3134.079–3137.510  my own LLMs?" Anyone [laughter] want to?
3137.510–3137.520  Anyone [laughter] want to?
3137.520–3139.510  Anyone [laughter] want to? >> There are lots of great tutorials. I
3139.510–3139.520  >> There are lots of great tutorials. I
3139.520–3141.910  >> There are lots of great tutorials. I want to say that like that tell how you
3141.910–3141.920  want to say that like that tell how you
3141.920–3143.109  want to say that like that tell how you can write
3143.109–3143.119  can write
3143.119–3145.349  can write >> like LLM training and inference like
3145.349–3145.359  >> like LLM training and inference like
3145.359–3149.030  >> like LLM training and inference like Andrew Corp's uh like tiny story will be
3149.030–3149.040  Andrew Corp's uh like tiny story will be
3149.040–3150.790  Andrew Corp's uh like tiny story will be like a good one if you want to get
3150.790–3150.800  like a good one if you want to get
3150.800–3153.510  like a good one if you want to get familiar with LLMs. But if you want to
3153.510–3153.520  familiar with LLMs. But if you want to
3153.520–3155.670  familiar with LLMs. But if you want to train your own [laughter] like I don't
3155.670–3155.680  train your own [laughter] like I don't
3155.680–3158.150  train your own [laughter] like I don't know uh
3158.150–3158.160  know uh
3158.160–3161.430  know uh leading model using just PyTorch you
3161.430–3161.440  leading model using just PyTorch you
3161.440–3163.109  leading model using just PyTorch you need a pretty substantial hardware
3163.109–3163.119  need a pretty substantial hardware
3163.119–3165.589  need a pretty substantial hardware investments uh which might be a bigger
3165.589–3165.599  investments uh which might be a bigger
3165.599–3166.470  investments uh which might be a bigger blocker
3166.470–3166.480  blocker
3166.480–3169.349  blocker >> and also you need a lot of data right so
3169.349–3169.359  >> and also you need a lot of data right so
3169.359–3172.470  >> and also you need a lot of data right so like I think the bottleneck is not your
3172.470–3172.480  like I think the bottleneck is not your
3172.480–3175.030  like I think the bottleneck is not your framework but your infrastructure and
3175.030–3175.040  framework but your infrastructure and
3175.040–3176.950  framework but your infrastructure and your data
3176.950–3176.960  your data
3176.960–3179.270  your data >> yeah I was going to say
3179.270–3179.280  >> yeah I was going to say
3179.280–3180.790  >> yeah I was going to say >> there there are some folks out there
3180.790–3180.800  >> there there are some folks out there
3180.800–3182.710  >> there there are some folks out there creating opensource models models that
3182.710–3182.720  creating opensource models models that
3182.720–3183.990  creating opensource models models that you could then maybe [laughter]
3183.990–3184.000  you could then maybe [laughter]
3184.000–3187.030  you could then maybe [laughter] refine. I don't know, Joe.
3187.030–3187.040  refine. I don't know, Joe.
3187.040–3190.390  refine. I don't know, Joe. >> Hopefully soon. Yeah, hopefully. Uh when
3190.390–3190.400  >> Hopefully soon. Yeah, hopefully. Uh when
3190.400–3192.069  >> Hopefully soon. Yeah, hopefully. Uh when those models models are hard to build, I
3192.069–3192.079  those models models are hard to build, I
3192.079–3195.190  those models models are hard to build, I have to say. Um no, I I I think like
3195.190–3195.200  have to say. Um no, I I I think like
3195.200–3196.549  have to say. Um no, I I I think like building models from scratch, like
3196.549–3196.559  building models from scratch, like
3196.559–3198.470  building models from scratch, like really good ones obviously is expensive.
3198.470–3198.480  really good ones obviously is expensive.
3198.480–3199.990  really good ones obviously is expensive. I think, you know, to Nikita's point,
3199.990–3200.000  I think, you know, to Nikita's point,
3200.000–3202.309  I think, you know, to Nikita's point, there's a definitely hardware is is a
3202.309–3202.319  there's a definitely hardware is is a
3202.319–3204.470  there's a definitely hardware is is a big challenge. I think there there's
3204.470–3204.480  big challenge. I think there there's
3204.480–3206.870  big challenge. I think there there's some really cool like I I'll I'll plug
3206.870–3206.880  some really cool like I I'll I'll plug
3206.880–3208.790  some really cool like I I'll I'll plug Unslo because like Daniel and Michael
3208.790–3208.800  Unslo because like Daniel and Michael
3208.800–3210.069  Unslo because like Daniel and Michael are really great and they have a great
3210.069–3210.079  are really great and they have a great
3210.079–3212.390  are really great and they have a great ecosystem. They have great tools. Uh you
3212.390–3212.400  ecosystem. They have great tools. Uh you
3212.400–3214.630  ecosystem. They have great tools. Uh you could basically pick out, you know, open
3214.630–3214.640  could basically pick out, you know, open
3214.640–3216.790  could basically pick out, you know, open source models and you can use unsloth
3216.790–3216.800  source models and you can use unsloth
3216.800–3220.309  source models and you can use unsloth and and and kind of like do RL and and
3220.309–3220.319  and and and kind of like do RL and and
3220.319–3221.910  and and and kind of like do RL and and SFT and and be able to do it really
3221.910–3221.920  SFT and and be able to do it really
3221.920–3224.549  SFT and and be able to do it really quick like quickly and cheaply. Uh even
3224.549–3224.559  quick like quickly and cheaply. Uh even
3224.559–3226.870  quick like quickly and cheaply. Uh even in notebooks um like there's actually a
3226.870–3226.880  in notebooks um like there's actually a
3226.880–3228.950  in notebooks um like there's actually a ton of examples there. So uh but like
3228.950–3228.960  ton of examples there. So uh but like
3228.960–3230.470  ton of examples there. So uh but like creating it from scratch is like quite
3230.470–3230.480  creating it from scratch is like quite
3230.480–3233.030  creating it from scratch is like quite expensive u quite involved. Uh we've
3233.030–3233.040  expensive u quite involved. Uh we've
3233.040–3235.109  expensive u quite involved. Uh we've kind of um we're we're at the point now
3235.109–3235.119  kind of um we're we're at the point now
3235.119–3236.710  kind of um we're we're at the point now as a community where it's just it's like
3236.710–3236.720  as a community where it's just it's like
3236.720–3239.030  as a community where it's just it's like prohibitively expensive to to do that as
3239.030–3239.040  prohibitively expensive to to do that as
3239.040–3241.109  prohibitively expensive to to do that as an individual. Um, but that said,
3241.109–3241.119  an individual. Um, but that said,
3241.119–3242.710  an individual. Um, but that said, there's a ton of great open models out
3242.710–3242.720  there's a ton of great open models out
3242.720–3244.230  there's a ton of great open models out there. Um, and you can play with them
3244.230–3244.240  there. Um, and you can play with them
3244.240–3245.670  there. Um, and you can play with them and you can customize them and you can
3245.670–3245.680  and you can customize them and you can
3245.680–3247.430  and you can customize them and you can run them locally now with Olana and
3247.430–3247.440  run them locally now with Olana and
3247.440–3249.510  run them locally now with Olana and other platforms. Um, so I think that's
3249.510–3249.520  other platforms. Um, so I think that's
3249.520–3252.069  other platforms. Um, so I think that's like um, yeah, obviously Karpath's uh,
3252.069–3252.079  like um, yeah, obviously Karpath's uh,
3252.079–3254.549  like um, yeah, obviously Karpath's uh, project is is super cool as well. U,
3254.549–3254.559  project is is super cool as well. U,
3254.559–3256.069  project is is super cool as well. U, there's a lot going on there. Um, but I
3256.069–3256.079  there's a lot going on there. Um, but I
3256.079–3257.109  there's a lot going on there. Um, but I wouldn't worry about training from
3257.109–3257.119  wouldn't worry about training from
3257.119–3258.150  wouldn't worry about training from scratch because there's just a ton to
3258.150–3258.160  scratch because there's just a ton to
3258.160–3259.510  scratch because there's just a ton to build on right now. It's just readily
3259.510–3259.520  build on right now. It's just readily
3259.520–3261.670  build on right now. It's just readily available.
3261.670–3261.680  available.
3261.680–3265.750  available. >> Yeah. Awesome. Uh, I think we had one or
3265.750–3265.760  >> Yeah. Awesome. Uh, I think we had one or
3265.760–3268.549  >> Yeah. Awesome. Uh, I think we had one or two more questions. Um so uh this one is
3268.549–3268.559  two more questions. Um so uh this one is
3268.559–3271.109  two more questions. Um so uh this one is from Manukumar.
3271.109–3271.119  from Manukumar.
3271.119–3273.030  from Manukumar. Uh
3273.030–3273.040  Uh
3273.040–3275.589  Uh are there any uh Google Summer of Code
3275.589–3275.599  are there any uh Google Summer of Code
3275.599–3278.950  are there any uh Google Summer of Code uh projects uh focused around PyTorch? I
3278.950–3278.960  uh projects uh focused around PyTorch? I
3278.960–3281.109  uh projects uh focused around PyTorch? I don't know if there are. Does anybody
3281.109–3281.119  don't know if there are. Does anybody
3281.119–3285.190  don't know if there are. Does anybody know among this group?
3285.190–3285.200  know among this group?
3285.200–3287.190  know among this group? >> I don't know. But that's a good idea.
3287.190–3287.200  >> I don't know. But that's a good idea.
3287.200–3290.710  >> I don't know. But that's a good idea. Like I think
3294.390–3294.400  maybe pitch a few few ideas to Google.
3294.400–3295.990  maybe pitch a few few ideas to Google. But like as name suggest, it's a Google
3295.990–3296.000  But like as name suggest, it's a Google
3296.000–3299.349  But like as name suggest, it's a Google summer of code. We don't run it. So if
3299.349–3299.359  summer of code. We don't run it. So if
3299.359–3301.030  summer of code. We don't run it. So if somebody wants to organize it, that
3301.030–3301.040  somebody wants to organize it, that
3301.040–3302.390  somebody wants to organize it, that sounds great. And like
3302.390–3302.400  sounds great. And like
3302.400–3302.870  sounds great. And like >> yeah,
3302.870–3302.880  >> yeah,
3302.880–3305.030  >> yeah, >> if somebody
3305.030–3305.040  >> if somebody
3305.040–3307.510  >> if somebody >> Yeah, we'll be happy to support.
3307.510–3307.520  >> Yeah, we'll be happy to support.
3307.520–3308.069  >> Yeah, we'll be happy to support. >> Yep.
3308.069–3308.079  >> Yep.
3308.079–3310.230  >> Yep. >> We should we should reach out to our uh
3310.230–3310.240  >> We should we should reach out to our uh
3310.240–3313.910  >> We should we should reach out to our uh our our Google um PyTorch Foundation. Uh
3313.910–3313.920  our our Google um PyTorch Foundation. Uh
3313.920–3314.309  our our Google um PyTorch Foundation. Uh uh
3314.309–3314.319  uh
3314.319–3315.109  uh >> there you go.
3315.109–3315.119  >> there you go.
3315.119–3318.630  >> there you go. >> And see what we can help.
3318.630–3318.640  >> And see what we can help.
3318.640–3322.069  >> And see what we can help. >> Exactly. Okay. Awesome. Um, Basil, were
3322.069–3322.079  >> Exactly. Okay. Awesome. Um, Basil, were
3322.079–3323.910  >> Exactly. Okay. Awesome. Um, Basil, were there any more questions in the in the
3323.910–3323.920  there any more questions in the in the
3323.920–3328.069  there any more questions in the in the comments? Oh, okay. Uh, no, I think
3328.069–3328.079  comments? Oh, okay. Uh, no, I think
3328.079–3329.510  comments? Oh, okay. Uh, no, I think Okay. Well, yeah, this is a slightly
3329.510–3329.520  Okay. Well, yeah, this is a slightly
3329.520–3330.870  Okay. Well, yeah, this is a slightly different question. With the silicon
3330.870–3330.880  different question. With the silicon
3330.880–3332.950  different question. With the silicon market exploding, how is PyTorch
3332.950–3332.960  market exploding, how is PyTorch
3332.960–3334.870  market exploding, how is PyTorch ensuring that the XPU, you know,
3334.870–3334.880  ensuring that the XPU, you know,
3334.880–3337.990  ensuring that the XPU, you know, torch.xpu can compile and run um models
3337.990–3338.000  torch.xpu can compile and run um models
3338.000–3340.870  torch.xpu can compile and run um models across Nvidia, AMD, and Intel chips with
3340.870–3340.880  across Nvidia, AMD, and Intel chips with
3340.880–3343.589  across Nvidia, AMD, and Intel chips with um zero performance loss.
3343.589–3343.599  um zero performance loss.
3343.599–3346.870  um zero performance loss. Um
3349.430–3349.440  >> I think it's a loaded question but also
3349.440–3351.349  >> I think it's a loaded question but also like uh it's very hard to define what
3351.349–3351.359  like uh it's very hard to define what
3351.359–3352.870  like uh it's very hard to define what zero component
3352.870–3352.880  zero component
3352.880–3353.589  zero component >> right
3353.589–3353.599  >> right
3353.599–3356.309  >> right >> don't think there is any like hardware
3356.309–3356.319  >> don't think there is any like hardware
3356.319–3359.670  >> don't think there is any like hardware spec that you can say well this this
3359.670–3359.680  spec that you can say well this this
3359.680–3361.510  spec that you can say well this this piece of silicon is exactly the same as
3361.510–3361.520  piece of silicon is exactly the same as
3361.520–3363.430  piece of silicon is exactly the same as another piece of silicon and also like a
3363.430–3363.440  another piece of silicon and also like a
3363.440–3364.549  another piece of silicon and also like a pure
3364.549–3364.559  pure
3364.559–3368.230  pure >> uh like a feature parity but as Andrew
3368.230–3368.240  >> uh like a feature parity but as Andrew
3368.240–3370.950  >> uh like a feature parity but as Andrew say there is CRC and there is in general
3370.950–3370.960  say there is CRC and there is in general
3370.960–3373.109  say there is CRC and there is in general and you should direct
3373.109–3373.119  and you should direct
3373.119–3375.030  and you should direct like project experience development
3375.030–3375.040  like project experience development
3375.040–3376.710  like project experience development partnership with Intel which is part of
3376.710–3376.720  partnership with Intel which is part of
3376.720–3379.750  partnership with Intel which is part of the PyTorch Foundation and they doing
3379.750–3379.760  the PyTorch Foundation and they doing
3379.760–3381.670  the PyTorch Foundation and they doing their best and community doing their
3381.670–3381.680  their best and community doing their
3381.680–3383.990  their best and community doing their best and what you can do to make sure
3383.990–3384.000  best and what you can do to make sure
3384.000–3385.670  best and what you can do to make sure that it is the case if you notice that
3385.670–3385.680  that it is the case if you notice that
3385.680–3388.470  that it is the case if you notice that something was working on your black fill
3388.470–3388.480  something was working on your black fill
3388.480–3390.789  something was working on your black fill and doesn't work on XPU please file an
3390.789–3390.799  and doesn't work on XPU please file an
3390.799–3392.390  and doesn't work on XPU please file an issue and I guess there will be people
3392.390–3392.400  issue and I guess there will be people
3392.400–3394.630  issue and I guess there will be people who happy to answer if this can be
3394.630–3394.640  who happy to answer if this can be
3394.640–3398.950  who happy to answer if this can be addressed in some way but yeah like
3398.950–3398.960  addressed in some way but yeah like
3398.960–3402.230  addressed in some way but yeah like we we want heterogeneity we want uh like
3402.230–3402.240  we we want heterogeneity we want uh like
3402.240–3404.390  we we want heterogeneity we want uh like models to work. We talked about torch
3404.390–3404.400  models to work. We talked about torch
3404.400–3406.710  models to work. We talked about torch accelerate. We talked about CRCR. We
3406.710–3406.720  accelerate. We talked about CRCR. We
3406.720–3408.789  accelerate. We talked about CRCR. We talked a lot about like a newer romcom.
3408.789–3408.799  talked a lot about like a newer romcom.
3408.799–3410.470  talked a lot about like a newer romcom. So there are lots of initiative but I
3410.470–3410.480  So there are lots of initiative but I
3410.480–3412.950  So there are lots of initiative but I guess zero performance loss is just
3412.950–3412.960  guess zero performance loss is just
3412.960–3415.510  guess zero performance loss is just unattainable and it's also I'm not sure
3415.510–3415.520  unattainable and it's also I'm not sure
3415.520–3416.470  unattainable and it's also I'm not sure what it means by
3416.470–3416.480  what it means by
3416.480–3418.150  what it means by >> not well defined. It's not well defined
3418.150–3418.160  >> not well defined. It's not well defined
3418.160–3420.069  >> not well defined. It's not well defined because different hardware performs you
3420.069–3420.079  because different hardware performs you
3420.079–3422.870  because different hardware performs you know just just behaves differently. Um
3422.870–3422.880  know just just behaves differently. Um
3422.880–3424.470  know just just behaves differently. Um and we try and give you lots of knobs
3424.470–3424.480  and we try and give you lots of knobs
3424.480–3427.109  and we try and give you lots of knobs and lots of capabilities but uh
3427.109–3427.119  and lots of capabilities but uh
3427.119–3428.630  and lots of capabilities but uh >> we try to reduce overhead with every
3428.630–3428.640  >> we try to reduce overhead with every
3428.640–3430.309  >> we try to reduce overhead with every release like the majority of the
3430.309–3430.319  release like the majority of the
3430.319–3431.510  release like the majority of the releases the features the kind of
3431.510–3431.520  releases the features the kind of
3431.520–3433.030  releases the features the kind of unspoken features that we don't talk
3433.030–3433.040  unspoken features that we don't talk
3433.040–3435.910  unspoken features that we don't talk about is we always try to bring a little
3435.910–3435.920  about is we always try to bring a little
3435.920–3439.829  about is we always try to bring a little bit of performance here and there.
3439.829–3439.839  bit of performance here and there.
3439.839–3441.829  bit of performance here and there. >> Yeah, just to add to it I think one of
3441.829–3441.839  >> Yeah, just to add to it I think one of
3441.839–3443.829  >> Yeah, just to add to it I think one of the interesting project maybe for you to
3443.829–3443.839  the interesting project maybe for you to
3443.839–3447.109  the interesting project maybe for you to look into is VLM. This is like they run
3447.109–3447.119  look into is VLM. This is like they run
3447.119–3449.270  look into is VLM. This is like they run multiple different models on different
3449.270–3449.280  multiple different models on different
3449.280–3452.150  multiple different models on different hardware and like as an example so you
3452.150–3452.160  hardware and like as an example so you
3452.160–3453.910  hardware and like as an example so you can explore it a little bit and I think
3453.910–3453.920  can explore it a little bit and I think
3453.920–3456.390  can explore it a little bit and I think it's open source and the CI is open
3456.390–3456.400  it's open source and the CI is open
3456.400–3460.309  it's open source and the CI is open source so um just as an example to look
3460.309–3460.319  source so um just as an example to look
3460.319–3464.230  source so um just as an example to look into could be interesting.
3464.230–3464.240  into could be interesting.
3464.240–3465.829  into could be interesting. >> Um I feel like we have answered this but
3465.829–3465.839  >> Um I feel like we have answered this but
3465.839–3468.150  >> Um I feel like we have answered this but I'll just quickly uh just just uh shout
3468.150–3468.160  I'll just quickly uh just just uh shout
3468.160–3469.910  I'll just quickly uh just just uh shout it out here. So somebody asked on the on
3469.910–3469.920  it out here. So somebody asked on the on
3469.920–3472.630  it out here. So somebody asked on the on the comment thread um is it uh is
3472.630–3472.640  the comment thread um is it uh is
3472.640–3475.190  the comment thread um is it uh is PyTorch supporting the newest CUDA? I
3475.190–3475.200  PyTorch supporting the newest CUDA? I
3475.200–3478.710  PyTorch supporting the newest CUDA? I think we're we're either on the newest
3478.710–3478.720  think we're we're either on the newest
3478.720–3480.549  think we're we're either on the newest CUDA or pretty close. And usually if you
3480.549–3480.559  CUDA or pretty close. And usually if you
3480.559–3482.390  CUDA or pretty close. And usually if you jump on nightlys, you can get the latest
3482.390–3482.400  jump on nightlys, you can get the latest
3482.400–3483.750  jump on nightlys, you can get the latest and greatest.
3483.750–3483.760  and greatest.
3483.760–3487.750  and greatest. >> Um, so Andre, you should your head when
3487.750–3487.760  >> Um, so Andre, you should your head when
3487.760–3488.870  >> Um, so Andre, you should your head when I said that. So I must have said
3488.870–3488.880  I said that. So I must have said
3488.880–3489.990  I said that. So I must have said something slightly wrong. So jump in
3489.990–3490.000  something slightly wrong. So jump in
3490.000–3490.630  something slightly wrong. So jump in here and correct me.
3490.630–3490.640  here and correct me.
3490.640–3494.150  here and correct me. >> So for 214, we're supporting up to CUDA
3494.150–3494.160  >> So for 214, we're supporting up to CUDA
3494.160–3497.190  >> So for 214, we're supporting up to CUDA 13.2. This is not the latest. The latest
3497.190–3497.200  13.2. This is not the latest. The latest
3497.200–3500.950  13.2. This is not the latest. The latest was um is 13.4 and it was released just
3500.950–3500.960  was um is 13.4 and it was released just
3500.960–3503.670  was um is 13.4 and it was released just like a week after we released PyTorch.
3503.670–3503.680  like a week after we released PyTorch.
3503.680–3505.589  like a week after we released PyTorch. So it's like we did not include it yet
3505.589–3505.599  So it's like we did not include it yet
3505.599–3507.349  So it's like we did not include it yet but it will be included in the next
3507.349–3507.359  but it will be included in the next
3507.359–3510.150  but it will be included in the next release of PyTorch and it is already in
3510.150–3510.160  release of PyTorch and it is already in
3510.160–3512.870  release of PyTorch and it is already in nightly and it has been in nightly for
3512.870–3512.880  nightly and it has been in nightly for
3512.880–3515.430  nightly and it has been in nightly for quite some time. So if you really want
3515.430–3515.440  quite some time. So if you really want
3515.440–3517.910  quite some time. So if you really want to have the latest CUDA download nightly
3517.910–3517.920  to have the latest CUDA download nightly
3517.920–3520.950  to have the latest CUDA download nightly and it should work uh for Linux only for
3520.950–3520.960  and it should work uh for Linux only for
3520.960–3523.430  and it should work uh for Linux only for Windows I think it's going to be enabled
3523.430–3523.440  Windows I think it's going to be enabled
3523.440–3526.069  Windows I think it's going to be enabled pretty soon maybe within one or two
3526.069–3526.079  pretty soon maybe within one or two
3526.079–3529.109  pretty soon maybe within one or two weeks.
3530.950–3530.960  >> Awesome. Well I think that was a good
3530.960–3533.589  >> Awesome. Well I think that was a good note to end on. Um, I think we had uh
3533.589–3533.599  note to end on. Um, I think we had uh
3533.599–3535.720  note to end on. Um, I think we had uh some fun conver fun conversation.
3535.720–3535.730  some fun conver fun conversation.
3535.730–3536.390  some fun conver fun conversation. [snorts]
3536.390–3536.400  [snorts]
3536.400–3538.390  [snorts] Um, a little a little repetition in the
3538.390–3538.400  Um, a little a little repetition in the
3538.400–3539.430  Um, a little a little repetition in the questions, but I think that's sort of
3539.430–3539.440  questions, but I think that's sort of
3539.440–3540.950  questions, but I think that's sort of built into the way we do things. So,
3540.950–3540.960  built into the way we do things. So,
3540.960–3543.829  built into the way we do things. So, thank you uh to my experts for uh for
3543.829–3543.839  thank you uh to my experts for uh for
3543.839–3547.109  thank you uh to my experts for uh for for putting up with that. Um, uh, so
3547.109–3547.119  for putting up with that. Um, uh, so
3547.119–3548.870  for putting up with that. Um, uh, so let's uh just in terms of closing
3548.870–3548.880  let's uh just in terms of closing
3548.880–3550.950  let's uh just in terms of closing remarks, um, thanks as always have to
3550.950–3550.960  remarks, um, thanks as always have to
3550.960–3552.470  remarks, um, thanks as always have to start with the contributors across the
3552.470–3552.480  start with the contributors across the
3552.480–3554.230  start with the contributors across the community who put in the steady effort
3554.230–3554.240  community who put in the steady effort
3554.240–3557.190  community who put in the steady effort that makes PyTorch what it is. So 500
3557.190–3557.200  that makes PyTorch what it is. So 500
3557.200–3560.150  that makes PyTorch what it is. So 500 odd people uh contributed to this uh
3560.150–3560.160  odd people uh contributed to this uh
3560.160–3561.910  odd people uh contributed to this uh particular release and it's been a
3561.910–3561.920  particular release and it's been a
3561.920–3563.670  particular release and it's been a number near that for the last several
3563.670–3563.680  number near that for the last several
3563.680–3565.910  number near that for the last several releases. Uh so it's just huge community
3565.910–3565.920  releases. Uh so it's just huge community
3565.920–3567.510  releases. Uh so it's just huge community effort. Uh and you know what we're
3567.510–3567.520  effort. Uh and you know what we're
3567.520–3569.589  effort. Uh and you know what we're talking about here is that effort. Uh so
3569.589–3569.599  talking about here is that effort. Uh so
3569.599–3571.430  talking about here is that effort. Uh so uh so thanks to all those folks. Uh
3571.430–3571.440  uh so thanks to all those folks. Uh
3571.440–3573.589  uh so thanks to all those folks. Uh thanks also to the three of you uh here
3573.589–3573.599  thanks also to the three of you uh here
3573.599–3575.670  thanks also to the three of you uh here on on on on stage answering these
3575.670–3575.680  on on on on stage answering these
3575.680–3577.829  on on on on stage answering these questions. Uh many of which you were
3577.829–3577.839  questions. Uh many of which you were
3577.839–3579.750  questions. Uh many of which you were answering un completely unprepared
3579.750–3579.760  answering un completely unprepared
3579.760–3581.030  answering un completely unprepared because they they were audience
3581.030–3581.040  because they they were audience
3581.040–3583.109  because they they were audience questions [laughter] in. So, thank you
3583.109–3583.119  questions [laughter] in. So, thank you
3583.119–3586.789  questions [laughter] in. So, thank you for being brave uh and wise. Um so, uh
3586.789–3586.799  for being brave uh and wise. Um so, uh
3586.799–3588.069  for being brave uh and wise. Um so, uh and being generous with your time and
3588.069–3588.079  and being generous with your time and
3588.079–3590.309  and being generous with your time and your expertise. Uh thanks also to behind
3590.309–3590.319  your expertise. Uh thanks also to behind
3590.319–3592.470  your expertise. Uh thanks also to behind the scenes uh Basil and Jyn who were
3592.470–3592.480  the scenes uh Basil and Jyn who were
3592.480–3594.870  the scenes uh Basil and Jyn who were scrambling to to get the uh the the
3594.870–3594.880  scrambling to to get the uh the the
3594.880–3597.190  scrambling to to get the uh the the questions uh from the from the
3597.190–3597.200  questions uh from the from the
3597.200–3599.910  questions uh from the from the discussion thread into uh into uh the
3599.910–3599.920  discussion thread into uh into uh the
3599.920–3601.190  discussion thread into uh into uh the screen here. So, thank you guys for
3601.190–3601.200  screen here. So, thank you guys for
3601.200–3603.349  screen here. So, thank you guys for working your magic. I appreciate that.
3603.349–3603.359  working your magic. I appreciate that.
3603.359–3605.430  working your magic. I appreciate that. Uh and then also thanks to everybody who
3605.430–3605.440  Uh and then also thanks to everybody who
3605.440–3606.870  Uh and then also thanks to everybody who joined and asked questions because you
3606.870–3606.880  joined and asked questions because you
3606.880–3609.510  joined and asked questions because you made this a good conversation. Uh we
3609.510–3609.520  made this a good conversation. Uh we
3609.520–3610.950  made this a good conversation. Uh we really appreciate it and keep building
3610.950–3610.960  really appreciate it and keep building
3610.960–3613.270  really appreciate it and keep building awesome things with PyTorch. Uh just a
3613.270–3613.280  awesome things with PyTorch. Uh just a
3613.280–3614.950  awesome things with PyTorch. Uh just a couple of resources uh for you to think
3614.950–3614.960  couple of resources uh for you to think
3614.960–3618.390  couple of resources uh for you to think about um a as we wrap up here. Um there
3618.390–3618.400  about um a as we wrap up here. Um there
3618.400–3621.430  about um a as we wrap up here. Um there are we released a blog with this release
3621.430–3621.440  are we released a blog with this release
3621.440–3623.030  are we released a blog with this release uh which has a lot of detailed uh
3623.030–3623.040  uh which has a lot of detailed uh
3623.040–3624.150  uh which has a lot of detailed uh details about the various different
3624.150–3624.160  details about the various different
3624.160–3625.430  details about the various different features. There's also the release
3625.430–3625.440  features. There's also the release
3625.440–3626.789  features. There's also the release notes. You should be able to find both
3626.789–3626.799  notes. You should be able to find both
3626.799–3628.789  notes. You should be able to find both of those really easily. We may drop
3628.789–3628.799  of those really easily. We may drop
3628.799–3630.069  of those really easily. We may drop links to them here. I'm not I'm not sure
3630.069–3630.079  links to them here. I'm not I'm not sure
3630.079–3632.230  links to them here. I'm not I'm not sure whether that will happen or not. Um also
3632.230–3632.240  whether that will happen or not. Um also
3632.240–3633.829  whether that will happen or not. Um also the developer communities. So
3633.829–3633.839  the developer communities. So
3633.839–3635.349  the developer communities. So discuss.pyarch.org
3635.349–3635.359  discuss.pyarch.org
3635.359–3637.190  discuss.pyarch.org and devdiscuss.pyrush.org
3637.190–3637.200  and devdiscuss.pyrush.org
3637.200–3639.589  and devdiscuss.pyrush.org are great uh for asking really detailed
3639.589–3639.599  are great uh for asking really detailed
3639.599–3641.910  are great uh for asking really detailed questions like you guys uh did here. Um
3641.910–3641.920  questions like you guys uh did here. Um
3641.920–3643.270  questions like you guys uh did here. Um and you'll you'll you'll be able to get
3643.270–3643.280  and you'll you'll you'll be able to get
3643.280–3645.910  and you'll you'll you'll be able to get the the the experts uh um across the
3645.910–3645.920  the the the experts uh um across the
3645.920–3648.069  the the the experts uh um across the community to answer. Um also there's
3648.069–3648.079  community to answer. Um also there's
3648.079–3650.150  community to answer. Um also there's some events coming up that I want to
3650.150–3650.160  some events coming up that I want to
3650.160–3652.309  some events coming up that I want to make people aware of. Uh so PyTorch
3652.309–3652.319  make people aware of. Uh so PyTorch
3652.319–3654.069  make people aware of. Uh so PyTorch Conference North America will be in San
3654.069–3654.079  Conference North America will be in San
3654.079–3655.990  Conference North America will be in San Jose in just a little bit more than a
3655.990–3656.000  Jose in just a little bit more than a
3656.000–3657.829  Jose in just a little bit more than a month. Um so that's October 20th and
3657.829–3657.839  month. Um so that's October 20th and
3657.839–3660.309  month. Um so that's October 20th and 21st. I think all of us will be here.
3660.309–3660.319  21st. I think all of us will be here.
3660.319–3662.549  21st. I think all of us will be here. All of us here will be will be there. Um
3662.549–3662.559  All of us here will be will be there. Um
3662.559–3665.829  All of us here will be will be there. Um and um uh you know it's a great chance
3665.829–3665.839  and um uh you know it's a great chance
3665.839–3668.549  and um uh you know it's a great chance to to to learn a ton about PyTorch and
3668.549–3668.559  to to to learn a ton about PyTorch and
3668.559–3671.349  to to to learn a ton about PyTorch and say hello to to to fellow uh uh members
3671.349–3671.359  say hello to to to fellow uh uh members
3671.359–3672.950  say hello to to to fellow uh uh members of the community. Um I'm certainly
3672.950–3672.960  of the community. Um I'm certainly
3672.960–3675.190  of the community. Um I'm certainly looking forward to it and if you're um
3675.190–3675.200  looking forward to it and if you're um
3675.200–3677.030  looking forward to it and if you're um part of the global audience um there are
3677.030–3677.040  part of the global audience um there are
3677.040–3678.710  part of the global audience um there are a bunch of PyTorch, you know, we've been
3678.710–3678.720  a bunch of PyTorch, you know, we've been
3678.720–3680.950  a bunch of PyTorch, you know, we've been really proliferating uh global events uh
3680.950–3680.960  really proliferating uh global events uh
3680.960–3682.710  really proliferating uh global events uh this year and I imagine that will
3682.710–3682.720  this year and I imagine that will
3682.720–3684.950  this year and I imagine that will continue. Uh so we had an event in Paris
3684.950–3684.960  continue. Uh so we had an event in Paris
3684.960–3686.950  continue. Uh so we had an event in Paris in the spring which was a lot of fun. We
3686.950–3686.960  in the spring which was a lot of fun. We
3686.960–3688.470  in the spring which was a lot of fun. We just completed events uh well a while
3688.470–3688.480  just completed events uh well a while
3688.480–3690.150  just completed events uh well a while ago in India and then more recently in
3690.150–3690.160  ago in India and then more recently in
3690.160–3692.150  ago in India and then more recently in China and I think we have one planned in
3692.150–3692.160  China and I think we have one planned in
3692.160–3695.589  China and I think we have one planned in Japan for later this year. So um and
3695.589–3695.599  Japan for later this year. So um and
3695.599–3697.430  Japan for later this year. So um and then I think there are pie days and
3697.430–3697.440  then I think there are pie days and
3697.440–3699.349  then I think there are pie days and smaller events happening all the time in
3699.349–3699.359  smaller events happening all the time in
3699.359–3701.589  smaller events happening all the time in lots of different places. So uh look
3701.589–3701.599  lots of different places. So uh look
3701.599–3704.710  lots of different places. So uh look look at the piece.org website for uh for
3704.710–3704.720  look at the piece.org website for uh for
3704.720–3707.190  look at the piece.org website for uh for u events near you. So I think that's it
3707.190–3707.200  u events near you. So I think that's it
3707.200–3709.670  u events near you. So I think that's it >> and last but no there is one more thing
3709.670–3709.680  >> and last but no there is one more thing
3709.680–3710.950  >> and last but no there is one more thing last but not least thank you very much
3710.950–3710.960  last but not least thank you very much
3710.960–3712.710  last but not least thank you very much Chris for being a fantastic moderator.
3712.710–3712.720  Chris for being a fantastic moderator.
3712.720–3713.990  Chris for being a fantastic moderator. We could not have wished for a better
3713.990–3714.000  We could not have wished for a better
3714.000–3715.750  We could not have wished for a better moderator here and without you it would
3715.750–3715.760  moderator here and without you it would
3715.760–3717.109  moderator here and without you it would be a very different event. So, thank
3717.109–3717.119  be a very different event. So, thank
3717.119–3717.990  be a very different event. So, thank you.
3717.990–3718.000  you.
3718.000–3720.390  you. >> Thank you, Chris. Great job.
3720.390–3720.400  >> Thank you, Chris. Great job.
3720.400–3720.950  >> Thank you, Chris. Great job. >> Bye-bye.
3720.950–3720.960  >> Bye-bye.
3720.960–3723.520  >> Bye-bye. >> Thank you.
