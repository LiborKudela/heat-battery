# What is this example
This example simulates transient hot wire (THW) method for measuring thermal
conductivity. It gives the voltage and current -> resistance of a ideal wire
that would appear on a connected Wheatstone bridge.
(see this [link](https://doi.org/10.1016/j.applthermaleng.2016.05.138)).
This simulations alows to assess what is the magnitude and evolution of the
voltage of interest, so apropriete electrical equipment can be purchased.

### Main idea of THW
The method is based on a fact that there is a correlation between the rate of 
change of the wire's temperature over time (more precisely the natural logarithm
of time) and thermal conductivity of the material that surrounds the wire. This
temperature change is induced with electrical current through the wire. The
temperature is calculated from the changing resistance of the wire itself.

![Single wire THW](assets/THW_01.jpg)

### Compensation for non-uniform temperature in the wire
After a electrical current is aplied to the wire of finite length a nonuniform 
temperature distribution along this wire is created.
The resistance reading of in the middle of the wire would be wrong due to this.

To compensate for this nonuniformity, two wires are used instead, one long and 
one short.

The wires will have very similar temperature profile near their ends.
We can look at the diference of the resistances of these two wires instead.
This will give us the resistance of an imaginary wire (the middle section
of the longer wire) that has almost uniform temperature distribution (because
there is a flat spot in the temperature profile in the middle of the long wire).

![Dual wire THW](assets/THW_02.jpg)

The resistance of this imaginary wire is than used to calculate the
thermal conductivity of the material. 

After this compensation there are very few things that affect the reading of
the thermal conductivity and very low uncertainty might be achieved.