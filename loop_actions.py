from compiler_gym.spaces.commandline import Commandline, CommandlineFlag

loop_opt_actions = [
        CommandlineFlag("-loop-deletion",        "-loop-deletion",             "-loop-deletion"),
        CommandlineFlag("-loop-distribute",      "-loop-distribute",           "-loop-distribute"),
        CommandlineFlag("-loop-fusion",          "-loop-fusion",               "-loop-fusion"),
        CommandlineFlag("-loop-guard-widening",  "-loop-guard-widening",       "-loop-guard-widening"),
        CommandlineFlag("-loop-idiom",           "-loop-idiom",                "-loop-idiom"),
        # CommandlineFlag("-loop-instsimplify",    "-loop-instsimplify",         "-loop-instsimplify"),
        CommandlineFlag("-loop-interchange",     "-loop-interchange",          "-loop-interchange"),
        CommandlineFlag("-loop-load-elim",       "-loop-load-elim",            "-loop-load-elim"),
        CommandlineFlag("-loop-predication",     "-loop-predication",          "-loop-predication"),
        CommandlineFlag("-loop-reroll",          "-loop-reroll",               "-loop-reroll"),
        CommandlineFlag("-loop-rotate",          "-loop-rotate",               "-loop-rotate"),
        # CommandlineFlag("-loop-simplifycfg",     "-loop-simplifycfg",          "-loop-simplifycfg"),
        # CommandlineFlag("-loop-simplify",        "-loop-simplify",             "-loop-simplify"),
        CommandlineFlag("-loop-sink",            "-loop-sink",                 "-loop-sink"),
        CommandlineFlag("-loop-reduce",          "-loop-reduce",               "-loop-reduce"),
        CommandlineFlag("-loop-unroll-and-jam",  "-loop-unroll-and-jam",       "-loop-unroll-and-jam"),
        CommandlineFlag("-loop-unroll",          "-loop-unroll",               "-loop-unroll"),
        CommandlineFlag("-loop-unswitch",        "-loop-unswitch",             "-loop-unswitch"),
        CommandlineFlag("-loop-vectorize",       "-loop-vectorize",            "-loop-vectorize"),
        CommandlineFlag("-loop-versioning-licm", "-loop-versioning-licm",      "-loop-versioning-licm"),
        CommandlineFlag("-loop-versioning",      "-loop-versioning",           "-loop-versioning"),
]

loop_action_space = Commandline(loop_opt_actions)