@testitem "Aqua" tags = [:aqua] begin
    using Aqua
    using Regress

    # `persistent_tasks` resolves Regress in a fresh throwaway project, which can only
    # draw on the General registry. Regress requires CovarianceMatrices 0.32, which is
    # not registered there, so the resolve fails before the check runs. Marking it
    # broken records the failure and reports it as unexpectedly passing once 0.32 is
    # registered, at which point this argument should go.
    Aqua.test_all(Regress; persistent_tasks = (broken = true,))
end
