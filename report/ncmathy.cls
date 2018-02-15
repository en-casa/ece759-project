\ProvidesClass{ncmathy}[]
\NeedsTeXFormat{LaTeX2e}

\LoadClass{article}

\RequirePackage[utf8]{inputenc}

% for clickable toc, add option ocgcolorlinks to print in correct color?
\RequirePackage[pdfusetitle]{hyperref}
\hypersetup{
    colorlinks,
    citecolor=black,
    filecolor=black,
    linkcolor=black,
    urlcolor=black
}

\RequirePackage{amsmath} % math mode
\RequirePackage{amssymb} % math mode
\RequirePackage[english]{babel}
\RequirePackage{cancel} % for diagonal strikethrough
\RequirePackage [autostyle, english = american]{csquotes}
\RequirePackage{fancyhdr} % fancy header
\RequirePackage{float}
\RequirePackage[margin=0.8in]{geometry}

% set document to have no indent
\setlength\parindent{0pt}

% set paragraph skip
\setlength{\parskip}{\baselineskip}

% to insert images
\RequirePackage{graphicx}
\graphicspath{ {images/}{../images/} } % location of source images, available to subfiles and main file

\RequirePackage{mathtools,xparse} % xsparse for if statements
\RequirePackage{multicol} % for multiple columns

% to print section name in header
\RequirePackage{nameref}
\makeatletter
	\newcommand*{\thesectiontitle}{\@currentlabelname}
\makeatother

\RequirePackage{subfiles} % to modularize tex files
\RequirePackage{stackrel} % to add extra lower text field

\numberwithin{equation}{section} % include sec. in eq. num.
\numberwithin{figure}{section} % include sec. in fig. num.
\numberwithin{table}{section} % include sec. in table. num.

%--------------------------------------------------------------------------
% section styling (depends on xcolor, calc, titlesec)
\RequirePackage{xcolor}
\RequirePackage{calc}
% for horizontal line under sections
\RequirePackage[explicit]{titlesec}

%Setting section title bg color
%\definecolor{redbg}{RGB}{255, 234, 234}
%Setting subsection and subsubsection title color
\definecolor{Red}{RGB}{123, 25, 25}

\titleformat{\section}
  {\normalfont\LARGE\bfseries\color{Red}}
  {\textcolor{Red}\thesection}
  {1em}
  %{\colorbox{redbg}{\parbox{\dimexpr\textwidth-2\fboxsep\relax}{\textcolor{Red}{\thesection\quad#1}}}}
  {\textcolor{Red}{#1}}
  [{
	\setlength{\titlewidth}{\textwidth}
	\titlerule[1pt]
  }]

%Underlining ruler for subsections
\titleformat{\subsection}
  {\normalfont\Large\bfseries\color{Red}}
  {\textcolor{Red}{\thesubsection}}
  {1em}
  {\textcolor{Red}{#1}}
  [{
  \setlength{\titlewidth}{0.8\textwidth}
  \titlerule[0.6pt]
  \setlength{\titlewidth}{\textwidth}
  }]

% subsubsections  
\titleformat{\subsubsection} 
  {\normalfont\normalsize\bfseries\color{Red}}
  {\textcolor{Red}{\thesubsubsection}}
  {1em}
  {\textcolor{Red}{#1}}
  [{
  \setlength{\titlewidth}{\widthof{\thesubsubsection\quad{#1}}}
  \titlerule[0.3pt]
  \setlength{\titlewidth}{\textwidth}
  }]
  
\let\oldtitleline\titleline
\renewcommand{\titleline}{\oldtitleline*}
  
%--------------------------------------------------------------------------
% title and header styles
% center title page on it's own page
\newcommand{\makeTitlePage}{
	\null  % Empty line
	\nointerlineskip  % No skip for prev line
	\vfill
	\let\snewpage \newpage
	\let\newpage \relax
	\maketitle
	\thispagestyle{empty} % remove page #
	\let \newpage \snewpage
	\vfill 
	\break % page break
}

% stylize title
\makeatletter
	\renewcommand{\title}[1]{\renewcommand{\@title}{\textcolor{\@titlecolor}{#1}}}
	\newcommand{\@titlecolor}{Red}
	\newcommand{\titlecolor}[1]{\renewcommand{\@titlecolor}{#1}}
\makeatother

% change toc title
\addto\captionsenglish{% Replace "english" with the language you use
  \renewcommand{\contentsname}%
    {Contents \footnotesize \textit{(Note that the entries are links.)}}
}

% stylize header
\pagestyle{fancy}
\fancyhf{}
\cfoot{\textcolor{Red}\thepage}
\chead{\textcolor{Red}\thesection :\ \textcolor{Red}\leftmark}
\rhead{\textcolor{Red}{ncasale}}
\lhead{\textcolor{Red}{\theassignment}}
	
\RequirePackage{etoolbox}
\makeatletter
\patchcmd{\f@nch@head}{\rlap}{\color{Red}\rlap}{}{}
\patchcmd{\headrule}{\hrule}{\color{Red}\hrule}{}{}
\patchcmd{\f@nch@foot}{\rlap}{\color{Red}\rlap}{}{}
\patchcmd{\footrule}{\hrule}{\color{Red}\hrule}{}{}
\makeatother

% header and footer **DEPRECATED**
\renewcommand{\sectionmark}[1]{\markboth{#1}{}} % set the \leftmark
\newcommand\setHeadFoot{
	\cfoot{\textcolor{Red}\thepage}
	\chead{\textcolor{Red}\thesection :\ \textcolor{Red}\leftmark}
	\rhead{\textcolor{Red}{ncasale}}
	\lhead{\textcolor{Red}{\theassignment}}
}

\fancypagestyle{fancytitle}{
    \fancyhf{} % clear page
    \cfoot{\textcolor{Red}\thepage}
    \renewcommand{\headrulewidth}{0pt}
}

\newcommand{\makeFancyTitle}{
	\maketitle
	\thispagestyle{fancytitle}
}
%--------------------------------------------------------------------------

% custom highlighting (depends on xcolor, soul) (for \code{.})
\RequirePackage{soul}
\newcommand{\hlc}[2][yellow]{{%
    \colorlet{foo}{#1}%
    \sethlcolor{foo}\hl{#2}}%
}
% use highlighting to identify code
\newcommand{\code}[1]{\hlc[pink!20]{\texttt{#1}}} 

% MATLAB code listing coloring
\RequirePackage[numbered,framed]{matlab-prettifier}
\RequirePackage{filecontents}
\RequirePackage[T1]{fontenc}
\let\ph\mlplaceholder % shorter macro
\lstMakeShortInline"

\lstset{
  style              = Matlab-editor,
  basicstyle         = \mlttfamily,
  escapechar         = ",
  mlshowsectionrules = true,
}

% c++ syntax highlighting (optional)
\newcommand{\cpp}{
	\lstset{language=C++,
                basicstyle=\ttfamily,
                keywordstyle=\color{blue}\ttfamily,
                stringstyle=\color{red}\ttfamily,
                commentstyle=\color{green}\ttfamily,
                morecomment=[l][\color{magenta}]{\#}
}}

%--------------------------------------------------------------------------
% math stuff

% pretty norm fences
\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\DeclarePairedDelimiter{\normgen}{\lVert}{\rVert} % inner norm, general
\newcommand{\norm}[2]{\normgen{#1}_{#2}}

% determinant (already defined)
\renewcommand{\det}[1]{\text{det}(#1)} 

% trace
\newcommand{\tr}[1]{\text{tr}(#1)}

%\NewDocumentCommand{\normL}{ s O{} m }{%
%  \IfBooleanTF{#1}{\norm*{#3}}{\norm[#2]{#3}}_{L_2(\Omega)}%
%}

% pretty inner products
\DeclarePairedDelimiterX{\inp}[2]{\langle}{\rangle}{#1 | #2}

% plus or minus (pom) symbol
\newcommand{\pom}{\sbox0{$1$}\sbox2{$\scriptstyle\pm$}
  \raise\dimexpr(\ht0-\ht2)/2\relax\box2 }
\newcommand{\pinus}{\pom}

% shorthand for inverse
\newcommand{\inv}{^{-1}}

% gaussian elimination arrows with spacing
\newcommand{\gaussElimSmall}[1]{\quad\stackrel{\mathclap{{#1}}}{\rightarrow}\quad}
\newcommand{\gaussElim}[1]{\quad\quad\stackrel{\mathclap{{#1}}}{\rightarrow}\quad\quad}
\newcommand{\gaussElimBig}[1]{\quad\quad\quad\stackrel{\mathclap{{#1}}}{\rightarrow}\quad\quad\quad}

% for augmented matrices
% \begin{amatrix}{m-1}, m cols
\newenvironment{amatrix}[1]{%
  \left(\begin{array}{@{}*{#1}{c}|c@{}}
}{%
  \end{array}\right)
}

% colored strike-through for cancelling values in equations
\newcommand\hcancel[2][red]{\renewcommand\CancelColor{\color{#1}}\cancel{#2}}

% normal distributions
\newcommand{\normnot}[2]{\mathcal{N}(#1,#2)} % provide mu, sigma
\newcommand{\normgeneric}{\normnot{\mu}{\sigma}} % generic
\newcommand{\normpdf}[2]{\frac{1}{{\sqrt{2\pi{#2}^2}}} \; e^{ -\frac{(x-{#1})^2}{(2){#2}^2} } }
\newcommand{\normpdfgeneric}{\normpdf{\mu}{\sigma}}

% natural Log
\newcommand{\lnn}[1]{\ln\left(#1\right)}

%--------------------------------------------------------------------------
% for vertical center in tables
\RequirePackage{makecell} % To keep spacing of text in tables
\setcellgapes{4pt} % parameter for the spacing
% some table formatting
\renewcommand\theadalign{cb}
\renewcommand\theadfont{\bfseries}
\renewcommand\theadgape{\Gape[4pt]}
\renewcommand\cellgape{\Gape[4pt]}

% skip to next column in multicols environment
\newcommand{\nextColumn}{\vfill\null \columnbreak}

% number any line of align* equations- just include \numberthis on the line
\newcommand\numberthis{\addtocounter{equation}{1}\tag{\theequation}}