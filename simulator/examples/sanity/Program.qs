// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

namespace Qir.Emission {

    open Microsoft.Quantum.Intrinsic;

    @EntryPoint()
    operation CustomFail() : Unit {
        fail "Custom Failure Message";
    }

    @EntryPoint()
    operation BasicTest() : Unit {
        let a = "Foo";
        let b = "Bar";
        let c = a + b;
        Message(c);
    }

    @EntryPoint()
    operation ConcatTest() : String {
        let a = "Hello";
        let b = ", World!";
        return a + b;
    }

    @EntryPoint()
    operation TupleTest() : Unit {
        let a = (1, "Bar");
        let b = ("Foo", 1);
        let (_, c) = a;
        let (d, _) = b;
        if d + c != "FooBar" {
            fail $"Tuple did not preserve expected values: {d + c}!";
        }
        Message("Finished Tuple test!");
    }

    @EntryPoint()
    operation RangeTest() : Unit {
        let array = [0, 1, 2, 3, 4, 5];
        let slice = array[0..2];
        let everyOther = array[0..2..5];
        let backwards = array[5..-1..0];
        if Length(array) != 6 {
            fail $"Original array had incorrect length: {Length(array)}";
        }
        if Length(slice) != 3 {
            fail $"Slice had incorrect length: {Length(slice)}, {slice}";
        }
        if slice[0] != 0 or slice[1] != 1 or slice[2] != 2 {
            fail $"Slice had incorrect contents: {slice}";
        }
        if Length(everyOther) != 3 {
            fail $"Slice with every other element had incorrect length: {Length(everyOther)}, {everyOther}";
        }
        if everyOther[0] != 0 or everyOther[1] != 2 or everyOther[2] != 4 {
            fail $"Slice with every other element had incorrect contents: {everyOther}";
        }
        if Length(backwards) != 6 {
            fail $"Reversed slice had incorrect length: {Length(backwards)}, {backwards}";
        }
        if backwards[0] != array[5] or backwards[1] != array[4] or backwards[2] != array[3] or
            backwards[3] != array[2] or backwards[4] != array[1] or backwards[5] != array[0] {
            fail $"Reversed slice had incorrect contents: {backwards} vs {array}";
        }
        Message("Finished Range test!");
    }

    operation SillyAssert(a : Int, b : Int) : Unit is Adj {
        body (...) {
            if a > b {
                fail $"{a} is not less than or equal {b}";
            }
        }
        adjoint (...) {
            if a < b {
                fail $"{a} is not greater than or equal {b}";
            }
        }
    }

    @EntryPoint()
    operation CallableTest(shouldFail : Bool) : Unit {
        SillyAssert(1, 1);
        let leq = SillyAssert;
        leq(1, 2);
        let geq = Adjoint SillyAssert;
        geq(2, 1);
        let assertNotNegative = SillyAssert(0, _);
        assertNotNegative(3);
        let geq42 = Adjoint SillyAssert(_, 42);
        geq42(43);
        if shouldFail {
            Adjoint geq42(43);
        } else {
            Adjoint geq42(41);
        }
        Message("Finished Callable test!");
    }

    @EntryPoint()
    operation ResultTest() : Unit {
        let one = One;
        let zero = Zero;
        if One != one {
            fail "Equivalence of literal One failed!";
        }
        if Zero != zero {
            fail "Equivalence of literal Zero failed!";
        }
        if one == zero {
            fail "Zero and One should not be equivalent!";
        }
        Message($"Literal One: {One}, Variable One: {one}");
        Message($"Literal Zero: {Zero}, Variable Zero: {zero}");
        Message("Finished Result test!");
    }
}