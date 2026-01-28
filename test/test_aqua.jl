@testitem "Aqua" tags = [:aqua] begin
    using Aqua
    using Regress

    Aqua.test_all(Regress)
end
